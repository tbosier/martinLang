//! A per-chain team of threads for the gradient, replacing the Martin
//! runtime's OpenMP-based mint_par_groups. Each chain thread owns a team of
//! `nt - 1` helper threads that live for the whole run and spin-wait on an
//! epoch counter (as libomp's workers spin between parallel regions), so a
//! gradient call costs no thread creation and no sleep/wake syscall.
//! Work split: groups q0..q1 = ngroups*k/nt .. ngroups*(k+1)/nt for thread k,
//! thread 0 being the caller, exactly as mint_par_groups splits them.
use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

pub type GroupFn = extern "C" fn(*mut c_void, i64, i64, i64);

#[derive(Clone, Copy)]
struct Job {
    f: Option<GroupFn>,
    ctx: usize,
    ngroups: i64,
    nt: i64,
}

struct Shared {
    epoch: AtomicU64,
    pending: AtomicI64,
    stop: AtomicBool,
    job: UnsafeCell<Job>,
}
// The job is written by the caller only while every helper is idle (pending
// == 0) and published by the Release increment of `epoch`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Team {
    size: usize,
    sh: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl Team {
    pub fn new(size: usize) -> Team {
        let sh = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            pending: AtomicI64::new(0),
            stop: AtomicBool::new(false),
            job: UnsafeCell::new(Job { f: None, ctx: 0, ngroups: 0, nt: 0 }),
        });
        let handles = (1..size)
            .map(|k| {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || helper(sh, k as i64))
            })
            .collect();
        Team { size, sh, handles }
    }

    fn run(&self, f: GroupFn, ctx: *mut c_void, ngroups: i64, nt: i64) -> i64 {
        let nt = nt.min(ngroups).min(self.size as i64);
        if nt <= 1 {
            f(ctx, 0, ngroups, 0);
            return 1;
        }
        unsafe { *self.sh.job.get() = Job { f: Some(f), ctx: ctx as usize, ngroups, nt } };
        self.sh.pending.store(self.size as i64 - 1, Ordering::Relaxed);
        self.sh.epoch.fetch_add(1, Ordering::Release);
        f(ctx, 0, ngroups / nt, 0);
        while self.sh.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        nt
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Release);
        for h in self.handles.drain(..) {
            h.join().ok();
        }
    }
}

fn helper(sh: Arc<Shared>, k: i64) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        loop {
            let e = sh.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if sh.stop.load(Ordering::Acquire) {
                return;
            }
            std::hint::spin_loop();
            spins += 1;
            if spins > 1 << 14 {
                std::thread::yield_now();
                spins = 0;
            }
        }
        let job = unsafe { *sh.job.get() };
        if k < job.nt {
            (job.f.unwrap())(job.ctx as *mut c_void, job.ngroups * k / job.nt, job.ngroups * (k + 1) / job.nt, k);
        }
        sh.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

thread_local! {
    static TEAM: RefCell<Option<Team>> = const { RefCell::new(None) };
    static THREADS: Cell<i64> = const { Cell::new(1) };
}

/// Sets the calling thread's gradient thread count (and builds its team).
pub fn set_threads(nt: i64) {
    THREADS.with(|t| t.set(nt.max(1)));
    TEAM.with(|t| {
        let mut t = t.borrow_mut();
        if t.as_ref().map(|x| x.size as i64) != Some(nt.max(1)) {
            *t = Some(Team::new(nt.max(1) as usize));
        }
    });
}

pub fn par_threads() -> i64 {
    THREADS.with(|t| t.get())
}

/// Same contract as the Martin runtime's mint_par_groups: runs f over
/// ngroups groups on up to nt threads (the caller is thread 0) and returns
/// the number of threads used.
pub fn par_groups(f: GroupFn, ctx: *mut c_void, ngroups: i64, nt: i64) -> i64 {
    TEAM.with(|t| {
        let mut t = t.borrow_mut();
        if t.as_ref().map_or(true, |x| (x.size as i64) < nt) {
            *t = Some(Team::new(nt.max(1) as usize));
        }
        t.as_ref().unwrap().run(f, ctx, ngroups, nt)
    })
}

"""Accuracy of Martin's vector log1p on [0, 1] (BernoulliLogit's softplus) as
the compiler emits it.

Extracts @mint_log1p01_v4 from the IR of a program whose model uses it,
links it into a C driver with clang, and compares log1p(e) (given
q = 1/(1 + e), as the kernel passes it) against long double log1pl:
uniform e in [0, 1], log-uniform e down to the smallest subnormal, dense e
around every table step, and 0, 1, the smallest subnormal and NaN.
usage: python3 tests/log/check_log1p.py IR_FILE [MAX_ULP, default 2]   (exit status 0 = pass)
"""
import os
import re
import subprocess
import sys
import tempfile

ir = open(sys.argv[1]).read()
max_ulp = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
keep = [l for l in ir.split("\n") if l.startswith("declare") or l.startswith("@mint_log1p_tab")]
m = re.search(r"define internal [^\n]*@mint_log1p01_v4\(.*?\n}\n", ir, re.S)
if not m:
    sys.exit(f"mint_log1p01_v4 not found in {sys.argv[1]}")
keep.append(m.group(0))
keep.append("""define void @vl4(ptr %a, ptr %b, ptr %c) {
  %e = load <4 x double>, ptr %a, align 8
  %q = load <4 x double>, ptr %b, align 8
  %y = call <4 x double> @mint_log1p01_v4(<4 x double> %e, <4 x double> %q)
  store <4 x double> %y, ptr %c, align 8
  ret void
}""")
driver = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
void vl4(const double *, const double *, double *);
static double ulps(double a, long double ref) {
  if (isnan(ref)) return isnan(a) ? 0 : 1e9;
  if (isnan(a)) return 1e9; /* a NaN where the reference is a number */
  double r = (double)ref;
  if (r == 0) return a == 0 ? 0 : 1e9;
  double u = nextafter(fabs(r), INFINITY) - fabs(r);
  return (double)(fabsl((long double)a - ref) / u);
}
static double worst = 0, wx = 0;
static long count = 0;
static double eb[4], qb[4];
static int nb = 0;
static void flush(void) {
  double out[4];
  for (int k = nb; k < 4; k++) eb[k] = 0.5, qb[k] = 1 / 1.5;
  vl4(eb, qb, out);
  for (int k = 0; k < nb; k++) {
    double u = ulps(out[k], log1pl((long double)eb[k]));
    if (u > worst) worst = u, wx = eb[k];
  }
  count += nb;
  nb = 0;
}
static void test(double e) { eb[nb] = e; qb[nb] = 1.0 / (1.0 + e); if (++nb == 4) flush(); }
int main(void) {
  srand48(19);
  for (long i = 0; i < 2000000; i++) test(drand48());
  for (long i = 0; i < 2000000; i++) test(exp2(-drand48() * 1074.0));
  for (int m = 0; m <= 256; m++) {
    /* e where round(256 (1 - 1/(1 + e))) changes: (1 + e) = 256/(256 - m + 0.5) */
    double c = 256.0 / (256.0 - m + 0.5) - 1.0;
    for (int d = -2000; d <= 2000; d++) { double e = c + d * 1e-12; if (e >= 0 && e <= 1) test(e); }
  }
  flush();
  double sp[] = {0.0, 1.0, 4.9406564584124654e-324, NAN};
  double qs[4], out[4];
  for (int k = 0; k < 4; k++) qs[k] = 1.0 / (1.0 + sp[k]);
  vl4(sp, qs, out);
  int bad = 0;
  for (int k = 0; k < 4; k++) {
    double y = out[k];
    long double ref = log1pl((long double)sp[k]);
    int ok = isnan(sp[k]) ? isnan(y) : ulps(y, ref) <= 0.5;
    if (!ok) { printf("special %g -> %.17g\n", sp[k], y); bad = 1; }
  }
  printf("worst %.3f ulp at %.17g over %ld inputs; special values %s\n", worst, wx, count, bad ? "WRONG" : "correct");
  return bad || worst > MAX_ULP;
}
"""
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
    open(f"{d}/l.ll", "w").write("\n".join(keep))
    open(f"{d}/d.c", "w").write(driver)
    subprocess.run(["clang", "-O2", "-march=native", f"-DMAX_ULP={max_ulp}", "-Wno-override-module", f"{d}/l.ll", f"{d}/d.c", "-o", f"{d}/t", "-lm"], check=True)
    r = subprocess.run([f"{d}/t"], capture_output=True, text=True)
    print("      " + r.stdout.strip().replace("\n", "\n      "))
    sys.exit(r.returncode)

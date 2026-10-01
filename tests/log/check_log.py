"""Accuracy of Mint's vector log as the compiler emits it.

Extracts @mint_log_v4 (and what it needs) from the IR of a program whose
model uses it, links it into a C driver with clang, and compares against
long double logl: random inputs over every binade of the double range
(log-uniform, including subnormals), dense inputs around 1 and around every
table boundary, and the special values 0, -0, negatives, subnormals, the
smallest and largest doubles, +-inf and NaN.
usage: python3 tests/log/check_log.py IR_FILE [MAX_ULP]   (exit status 0 = pass)
"""
import os
import re
import subprocess
import sys
import tempfile

ir = open(sys.argv[1]).read()
max_ulp = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
keep = [l for l in ir.split("\n") if l.startswith("declare") or l.startswith("@mint_log_")]
for name in ["mint_log_full_v4", "mint_log_v4"]:
    m = re.search(r"define internal [^\n]*@" + name + r"\(.*?\n}\n", ir, re.S)
    if not m:
        sys.exit(f"{name} not found in {sys.argv[1]}")
    keep.append(m.group(0))
keep.append("""define void @vlog4(ptr %a, ptr %b) {
  %x = load <4 x double>, ptr %a, align 8
  %y = call <4 x double> @mint_log_v4(<4 x double> %x)
  store <4 x double> %y, ptr %b, align 8
  ret void
}""")
driver = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void vlog4(const double *, double *);
static double ulps(double a, long double ref) {
  if (isnan(ref)) return isnan(a) ? 0 : 1e9;
  if (isnan(a)) return 1e9; /* a NaN where the reference is a number */
  if (isinf(ref)) return a == ref ? 0 : 1e9;
  double r = (double)ref;
  if (r == 0) return a == 0 ? 0 : 1e9;
  double u = nextafter(fabs(r), INFINITY) - fabs(r);
  return (double)(fabsl((long double)a - ref) / u);
}
static double from_bits(uint64_t b) { double x; memcpy(&x, &b, 8); return x; }
static double worst = 0, wx = 0;
static long count = 0;
static double buf[4];
static int nb = 0;
static void flush(void) {
  double out[4];
  for (int k = nb; k < 4; k++) buf[k] = 1.0;
  vlog4(buf, out);
  for (int k = 0; k < nb; k++) {
    double u = ulps(out[k], logl((long double)buf[k]));
    if (u > worst) worst = u, wx = buf[k];
  }
  count += nb;
  nb = 0;
}
static void test(double x) { buf[nb++] = x; if (nb == 4) flush(); }
int main(void) {
  srand48(13);
  /* log-uniform over the whole positive range, subnormals included */
  for (long i = 0; i < 2000000; i++) test(exp2(drand48() * 2098.0 - 1074.0));
  /* uniform near 1 and in [0.5, 2) */
  for (long i = 0; i < 1000000; i++) test(1.0 + (drand48() * 2 - 1) * 0.02);
  for (long i = 0; i < 1000000; i++) test(0.5 + drand48() * 1.5);
  for (long i = 0; i < 200000; i++) test(1.0 + (drand48() * 2 - 1) * 1e-10);
  /* both sides of every table boundary, in several binades */
  for (int e = -3; e <= 3; e++)
    for (uint64_t j = 0; j <= 128; j++) {
      uint64_t b = 0x3FE5F00000000000ULL + (j << 45) + ((uint64_t)(int64_t)e << 52);
      for (int d = -40; d <= 40; d++) test(from_bits(b + d));
    }
  flush();
  double sp[] = {0.0, -0.0, -1.0, -1e-300, -INFINITY, INFINITY, NAN, -NAN,
                 4.9406564584124654e-324, 2.2250738585072009e-308, 2.2250738585072014e-308, 1.7976931348623157e308,
                 1.0, 2.0, 0.5, 1e-310, 3e-320, 7.0, 1.0000000000000002, 0.99999999999999989};
  int bad = 0;
  for (int i = 0; i < 20; i += 4) {
    double out[4];
    vlog4(sp + i, out);
    for (int k = 0; k < 4; k++) {
      double x = sp[i + k], y = out[k], ref = log(x);
      int ok = isnan(ref) ? isnan(y) : (y == ref || ulps(y, logl((long double)x)) <= 1.0);
      if (!ok) { printf("special %g -> %.17g (libm %.17g)\n", x, y, ref); bad = 1; }
    }
  }
  printf("worst %.3f ulp at %.17g over %ld inputs; special values %s\n", worst, wx, count, bad ? "WRONG" : "match libm");
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

"""Accuracy of Martin's vector exp as the compiler emits it.

Extracts @mint_exp_v4 (and what it needs) from the IR of a program whose
model uses it, links it into a C driver with clang, and compares against
long double expl over random inputs across the whole range, plus NaN,
+-inf, -0, the overflow and underflow thresholds and subnormal results.
usage: python3 tests/exp/check_exp.py IR_FILE   (exit status 0 = pass)
"""
import os
import re
import subprocess
import sys
import tempfile

ir = open(sys.argv[1]).read()
keep = [l for l in ir.split("\n") if l.startswith("declare") or l.startswith("@mint_exp_tab")]
for name in ["mint_exp_full_v4", "mint_exp_v4"]:
    m = re.search(r"define internal [^\n]*@" + name + r"\(.*?\n}\n", ir, re.S)
    if not m:
        sys.exit(f"{name} not found in {sys.argv[1]}")
    keep.append(m.group(0))
keep.append("""define void @vexp4(ptr %a, ptr %b) {
  %x = load <4 x double>, ptr %a, align 8
  %y = call <4 x double> @mint_exp_v4(<4 x double> %x)
  store <4 x double> %y, ptr %b, align 8
  ret void
}""")
driver = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
void vexp4(const double *, double *);
static double ulps(double a, double ref) {
  if (isnan(ref)) return isnan(a) ? 0 : 1e9;
  if (isinf(ref)) return a == ref ? 0 : 1e9;
  if (ref == 0) return a == 0 ? 0 : 1e9;
  double u = nextafter(fabs(ref), INFINITY) - fabs(ref);
  return fabs(a - ref) / u;
}
int main(void) {
  double a[4], b[4], worst = 0, wx = 0;
  srand48(11);
  for (long i = 0; i < 3000000; i += 4) {
    for (int k = 0; k < 4; k++) {
      long j = i + k;
      a[k] = j % 3 == 0 ? (drand48() * 2 - 1) * 20 : j % 3 == 1 ? (drand48() * 2 - 1) * 750 : (drand48() * 2 - 1) * 0.5;
    }
    vexp4(a, b);
    for (int k = 0; k < 4; k++) {
      double u = ulps(b[k], (double)expl((long double)a[k]));
      if (u > worst) worst = u, wx = a[k];
    }
  }
  double sp[] = {0.0, -0.0, 1.0, -1.0, NAN, INFINITY, -INFINITY, 709.78, 709.79, 708.1, -708.1,
                 -708.4, -740.0, -745.1, -746.0, 1e-300, -1e-300, 707.9, -707.9, 1e3};
  int bad = 0;
  for (int i = 0; i < 20; i += 4) {
    vexp4(sp + i, b);
    for (int k = 0; k < 4; k++) {
      double u = ulps(b[k], (double)expl((long double)sp[i + k]));
      if (u > 2) { printf("special %g -> %.17g\n", sp[i + k], b[k]); bad = 1; }
    }
  }
  printf("worst %.2f ulp at %.17g\n", worst, wx);
  return bad || worst > 2.0;
}
"""
with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
    open(f"{d}/e.ll", "w").write("\n".join(keep))
    open(f"{d}/d.c", "w").write(driver)
    subprocess.run(["clang", "-O2", "-march=native", "-Wno-override-module", f"{d}/e.ll", f"{d}/d.c", "-o", f"{d}/t", "-lm"], check=True)
    r = subprocess.run([f"{d}/t"], capture_output=True, text=True)
    print("      " + r.stdout.strip().replace("\n", "\n      "))
    sys.exit(r.returncode)

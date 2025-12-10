import os
import subprocess
import re

try:
    os.remove("./Part_2/stats.csv")
except OSError:
    pass

threads_per_block = [1, 32, 64, 128, 256]
N = [2, 4, 6, 8, 10, 12]
M = [1, 3, 7, 9, 11]

repeats = range(0,10)
executables = ['matrix_sequential', 'matrix_cuda_gpu', 'matrix_cuda_shared_memory', 'matrix_cuda_2_level_reduction', 'matrix_cuda_shared_memory_optimized']


for n in N:
    for m in M:
        for tpb in threads_per_block:
            for repeat in repeats:
                for executable in executables:
                    args = ("./exe_bin/tp_cuda_part_2_" + executable, "-N", str(n), "-M", str(m), "-T", str(tpb))
                    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
                    popen.wait()

                    try:
                        out, err = popen.communicate(timeout=1)
                    except Exception:
                        out = b''
                    if isinstance(out, bytes):
                        out = out.decode(errors='ignore')

                    # Extract time from output
                    regex = re.search(r'in\s+([0-9.+-eE]+)\s+seconds', out)
                    if regex:
                        time_s = regex.group(1)
                    else:
                        print(f"Warning: Could not extract pi and time from output for executable={executable}, n={n}, m={m}, tpb={tpb}. Output was:\n{out}")

                    csv_path = './Part_2/stats.csv'
                    with open(csv_path, 'a') as fh:
                        fh.write(f"{executable},{n},{m},{tpb},{time_s}\n")
                        print(f"Wrote to {csv_path}: {executable},{n},{m},{tpb},{time_s}")


import os
import subprocess
import re

try:
    os.remove("./Part_3/stats.csv")
except OSError:
    pass

# Matrix dimensions as specified in the PDF
# N, M, P: 1000, 4000, 8000, 12000, 18000
dimensions = [1000, 4000, 8000, 12000, 18000]

repeats = range(0, 10)
executables = [
    'matrix_mult_sequential',
    'matrix_mult_cuda_1thread',
    'matrix_mult_cuda_shared',
    'matrix_mult_cuda_float',
    'matrix_mult_cuda_half'
]

for n in dimensions:
    for m in dimensions:
        for p in dimensions:
            for repeat in repeats:
                for executable in executables:
                    print(f"Running {executable} with N={n}, M={m}, P={p}, repeat={repeat}")
                    
                    args = ("./exe_bin/tp_cuda_part_3_" + executable, "-N", str(n), "-M", str(m), "-P", str(p))
                    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    try:
                        out, err = popen.communicate(timeout=300)  # 5 minute timeout
                    except subprocess.TimeoutExpired:
                        popen.kill()
                        out, err = popen.communicate()
                        print(f"  TIMEOUT for {executable} with N={n}, M={m}, P={p}")
                        time_s = 'TIMEOUT'
                    except Exception as e:
                        print(f"  ERROR for {executable} with N={n}, M={m}, P={p}: {e}")
                        out = b''
                        time_s = 'ERROR'
                    
                    if isinstance(out, bytes):
                        out = out.decode(errors='ignore')
                    
                    if time_s not in ['TIMEOUT', 'ERROR']:
                        # Extract time from output
                        # Looking for pattern: " N 1000 M 1000 P 1000 multiplication in 0.123456 seconds"
                        regex = re.search(r'multiplication in\s+([0-9.+-eE]+)\s+seconds', out)
                        if regex:
                            time_s = regex.group(1)
                        else:
                            print(f"  Warning: Could not extract time from output for executable={executable}, N={n}, M={m}, P={p}")
                            print(f"  Output was:\n{out}")
                            time_s = 'PARSE_ERROR'
                    
                    csv_path = './Part_3/stats.csv'
                    with open(csv_path, 'a') as fh:
                        fh.write(f"{executable},{n},{m},{p},{time_s}\n")
                        print(f"  Wrote to {csv_path}: {executable},{n},{m},{p},{time_s}")

print("\nBenchmarking complete! Results saved to Part_3/stats.csv")

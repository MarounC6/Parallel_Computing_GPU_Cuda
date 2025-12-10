import os
import subprocess
import re

try:
    os.remove("./Part_1/stats.csv")
except OSError:
    pass

num_steps = [1000, 1000000]#, 100000000, 10000000000, 1000000000000]
threads_per_block = [1, 32, 64, 128, 256]

repeats = range(0,10)
executables = ['pi_sequential', 'pi_cuda_gpu', 'pi_cuda_shared_memory', 'pi_cuda_2_level_reduction', 'pi_cuda_gpu_tableau']#, 'pi_cuda_tableau_2_level_reduction', 'pi_multistage_reduction']

for nsteps in num_steps:
    for tpb in threads_per_block:
        for repeat in repeats:
            for executable in executables:
                args = ("./exe_bin/tp_cuda_part_1_" + executable, "-N", str(nsteps), "-T", str(tpb))
                popen = subprocess.Popen(args, stdout=subprocess.PIPE)
                popen.wait()

                try:
                    out, err = popen.communicate(timeout=1)
                except Exception:
                    out = b''
                if isinstance(out, bytes):
                    out = out.decode(errors='ignore')

                # Extract pi value and time from output
                m = re.search(r'pi with\s+(\d+)\s+steps is\s+([0-9.+-eE]+)\s+in\s+([0-9.+-eE]+)\s+seconds', out)
                if m:
                    pi_val = m.group(2)
                    time_s = m.group(3)
                else:
                    # fallback: use last non-empty line of stdout
                    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                    last = lines[-1] if lines else ''
                    pi_val = ''
                    time_s = ''
                    # try to extract numbers from last line if possible
                    m2 = re.search(r'([0-9.+-eE]+).*in.*([0-9.+-eE]+)\s*seconds', last)
                    if m2:
                        pi_val = m2.group(1)
                        time_s = m2.group(2)
                    else:
                        print(f"Warning: Could not extract pi and time from output for executable={executable}, nsteps={nsteps}, tpb={tpb}. Output was:\n{out}")

                csv_path = './Part_1/stats.csv'
                with open(csv_path, 'a') as fh:
                    fh.write(f"{executable},{nsteps},{tpb},{time_s}\n")
                    print(f"Wrote to {csv_path}: {executable},{nsteps},{tpb},{time_s}")


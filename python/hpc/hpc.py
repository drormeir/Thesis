import warnings
import gc
import numpy as np
from types import SimpleNamespace

globals = SimpleNamespace(
    numba_import_success = False,
    cpu_njit_num_threads = np.uint32(0),
    cuda_available = False,
    max_threads_per_block = np.uint32(1),
    min_grid_size = np.uint32(1),
    warp_size = np.uint32(32),
    grid_block_shape_debug = np.uint32(0),
    max_registers_per_block = np.uint32(1),
    cuda_import_success = None
)

import subprocess, tempfile, os, re, errno

def search_ptx_version_error(str_stderr: str, verbose: bool = True) -> tuple[str,str]:
    ptx_match = re.search(r"Unsupported .version (\d+\.\d+); current version is '(\d+\.\d+)'", str_stderr)
    if not ptx_match:
        return '',''
    ptx = ptx_match.group(1).strip()
    max_ptx = ptx_match.group(2).strip()
    if verbose:
        print(f"PTX {ptx} unsupported, max supported is {max_ptx}", flush=True)
    return ptx, max_ptx



#####################################################

import subprocess
import sys
import os
import re
import threading


def get_nvidia_smi_cuda_version() -> str:
    ret = ''
    try:
        # Run nvidia-smi and capture output
        err, output = run_command(['nvidia-smi'])
        if err:
            print(f"Error executing nvidia-smi:\n{output}")
            return ret
        print(output)   
        # Look for "CUDA Version: X.Y" in the output
        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', output)
        if match:
            ret =  match.group(1)  # Returns version like "12.4"
        else:
            print("CUDA version not found in nvidia-smi output")
    except subprocess.CalledProcessError:
        print("Error: nvidia-smi command failed. Is it installed and accessible?")
    except Exception as e:
        print(f"Error parsing nvidia-smi: {e}")
    return ret


def get_nvcc_cuda_version() -> str:
    print('Checking nvcc --version...')
    ret = ''
    try:
        err, output = run_command(['nvcc', '--version'])
        if err:
            print(f"Error executing nvcc:\n{output}")
            return ret
        print(output)
        # Look for "release X.Y" in the output
        match = re.search(r'release\s+(\d+\.\d+)', output, re.IGNORECASE)
        if match:
            ret = match.group(1)  # Returns version like "12.5"
        else:
            print("CUDA version not found in nvcc output")
    except subprocess.CalledProcessError:
        print("Error: nvcc command failed. Is CUDA Toolkit installed and in PATH?")
    except Exception as e:
        print(f"Error parsing nvcc: {e}")
    return ret


def run_dummy_kernel(grid_size: int, block_dim: int) -> tuple[int, str]:
    """
    Execute dummy_numba_kernel.py with given grid size and block dimension.
    Returns: (exit_code, stdout, stderr)
    """
    python_path = sys.executable
    script_path = os.path.join("python","hpc", "dummy_numba_kernel.py")
    return run_command([python_path, script_path, str(grid_size), str(block_dim)])


def run_command(command: list[str], sudo: bool=False, stream_output: bool=False, timeout=None) -> tuple[int, str]:
    """
    Run a shell command, optionally with sudo.
    
    If stream_output is True, prints the command's output immediately as it's produced
    while capturing both stdout and stderr.
    
    Returns:
        tuple[int, str]: (errno, output)
            errno: 0 for success, or standard errno value for errors:
                  EPERM(1): Operation not permitted (command failed)
                  ENOENT(2): No such file or directory (command not found)
                  EACCES(13): Permission denied
                  EIO(5): Input/output error (general command failure)
                  ETIMEDOUT(110): Connection timed out
            output: Command output or error message
    """
    str_command_4_print = ' '.join(command)
    if sudo and command[0] != 'sudo':
        command = ['sudo'] + command

    if not stream_output:
        try:
            proc = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
            output = proc.stdout.strip()
            stderr_output = proc.stderr.strip()
        except subprocess.TimeoutExpired as e:
            return_message = f"Command '{str_command_4_print}' timed out after {timeout} seconds."
            if e.stdout:
                return_message += f'\nPartial output:\n{e.stdout.strip()}'
            if e.stderr:
                return_message += f'\nError message:\n{e.stderr.strip()}'
            return errno.ETIMEDOUT, return_message
        except subprocess.CalledProcessError as e:
            return_message = f"Command '{str_command_4_print}' failed with error code {e.returncode}"
            if e.stdout:
                return_message += f'\nPartial output:\n{e.stdout.strip()}'
            if e.stderr:
                return_message += f"\nError message: {e.stderr.strip()}"
            return errno.EIO, return_message
        except Exception as e:
            if isinstance(e, OSError):
                return e.errno, f"System error executing '{str_command_4_print}': {e}"
            return errno.EPERM, f"System error executing '{str_command_4_print}': {e}"
    else:
        print(f'Executing shell command: {str_command_4_print}')
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                return errno.ENOENT, f"Command not found: '{str_command_4_print}'"
            if isinstance(e, PermissionError):
                return errno.EACCES, f"Permission denied: '{str_command_4_print}'"
            if isinstance(e, OSError):
                return e.errno, f"Failed to start '{str_command_4_print}': {e}"
            return errno.EPERM, f"Failed to start '{str_command_4_print}': {e}"

        stdout_lines = []
        stderr_lines = []

        def read_stream(stream, output_lines):
            try:
                for line in iter(stream.readline, ''):
                    print(line, end='', flush=True)  # Print immediately
                    output_lines.append(line)
            finally:
                stream.close()

        stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines))
        stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines))
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_thread.join(1)  # Give threads 1 second to finish
            stderr_thread.join(1)
            return_message = f"Command '{str_command_4_print}' timed out after {timeout} seconds."
            if stdout_lines:
                return_message += '\nPartial output:\n' + '\n'.join(stdout_lines)
            if stderr_lines:
                return_message += '\nError message:\n' + '\n'.join(stderr_lines)
            return errno.ETIMEDOUT, return_message

        stdout_thread.join()
        stderr_thread.join()
        output = '\n'.join(stdout_lines)
        stderr_output = '\n'.join(stderr_lines)

        if returncode != 0:
            return_message = f"Command '{str_command_4_print}' failed with code {returncode}"
            if output:
                return_message += '\nPartial output:\n' + output
            if stderr_output:
                return_message += '\nError message:\n' + stderr_output
            return errno.EIO, return_message

    if stderr_output:
        if output:
            output += '\n'
        output += 'Error message:\n' + stderr_output

    return os.EX_OK, output  # Standard for successful execution (0)


def detect_cuda_compatibility_before_any_import(sudo: bool = False) -> bool:
    print('Detecting CUDA version prior to importing numba...')
    nvcc_version = get_nvcc_cuda_version()
    smi_version = get_nvidia_smi_cuda_version()
    if not nvcc_version or not smi_version:
        return False
    print(f'Found: nvcc version: {nvcc_version}   nvidia-smi cuda version: {smi_version} ')
    if float(nvcc_version) < float(smi_version) + 1e-6:
        return True
    print('nvcc version is newer than nvidia-smi cuda version --> trying to downgrading cuda-toolkit...')
    
    run_command(['ls','-l','/usr/local/cuda*'], stream_output=True)
    if not run_command(['apt-get', 'remove', '-y', 'cuda'], stream_output=True, sudo=sudo): # Remove CUDA metapackage
        print("Uninstallation failed. Aborting.")
        return False
    cuda_package = f"cuda-{smi_version.replace('.', '-')}"
    if not run_command(['apt-get', 'install', '-y', cuda_package], stream_output=True, sudo=sudo):
        print(f"Failed to install CUDA {smi_version}. Aborting.")
        return False
    print(f"Successfully downgraded CUDA Toolkit to {smi_version}.")
    exit_code, str_output = run_dummy_kernel(1,1)
    if exit_code == 0:
        return True
    print(f'run_dummy_kernel(1,1) --> {exit_code=}\n{str_output}')
    search_ptx_version_error(str_output)
    return False


if not detect_cuda_compatibility_before_any_import():
    globals.cuda_import_success = False

try:
    import numba
    print(f"Numba version: {numba.__version__}", flush=True)
    globals.numba_import_success = True
except ImportError as e:
    print(f"Could not import Numba.\n{e}", flush=True)

def ensure_cuda_available_or_explain():
    import os, sys, ctypes, pathlib, subprocess, textwrap
    try:
        from numba import cuda
    except Exception as e:
        raise RuntimeError("Numba CUDA import failed. Try: pip install -U numba numba-cuda cuda-python\n"
                           f"Import error: {e}") from e
    if cuda.is_available():
        return
    msgs = []
    for lib in ("libcuda.so.1", "libcuda.so"):
        try:
            ctypes.CDLL(lib); msgs.append(f"Loaded {lib}")
            break
        except OSError as e:
            msgs.append(f"Failed to load {lib}: {e}")
    devs = ["/dev/nvidia0","/dev/nvidiactl","/dev/nvidia-uvm","/dev/nvidia-uvm-tools"]
    missing = [d for d in devs if not pathlib.Path(d).exists()]
    if missing:
        msgs.append("Missing device nodes: " + ", ".join(missing))
    try:
        dmesg = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=1).stdout.lower().splitlines()
        tail = [ln for ln in dmesg if any(k in ln for k in ("nvidia","uvm","xid","rm","modeset"))][-20:]
        if tail:
            msgs.append("Kernel messages:\n" + "\n".join(tail))
    except Exception:
        pass
    hints = [
        "Restart services: sudo systemctl restart nvidia-persistenced",
        "Reload UVM: sudo rmmod nvidia_uvm || true && sudo modprobe nvidia_uvm",
        "If nodes missing: sudo apt install -y nvidia-modprobe && sudo nvidia-modprobe -u -c=0",
        "Optional: pip install -U numba-cuda cuda-python for newer CUDA support."
    ]
    raise RuntimeError("CUDA isn’t available (driver init failed).\n\nDiagnostics:\n  "
                       + "\n  ".join(msgs or ["<none>"])
                       + "\n\nHow to fix:\n  " + "\n  ".join(hints))


if globals.cuda_import_success is None:
    globals.cuda_import_success = False
    if globals.numba_import_success:
        try:
            import numba.cuda
            from numba.cuda.cudadrv.devicearray import DeviceNDArray
            from numba.core.errors import NumbaPerformanceWarning
            from numba.cuda.cudadrv.error import CudaSupportError
            ensure_cuda_available_or_explain()
            globals.cuda_import_success = True
            print('import numba.cuda --> SUCCESS!', flush=True)
        except ImportError as e:
            print(f'Could not import CUDA.\n{e}', flush=True)
        except Exception as e:
            print(f'{e}', flush=True)

###########################################################################################


def init_njit() -> np.uint32:
    if not globals.numba_import_success:
        return np.uint32(0)
    try:
        # Attempt to define and call a simple njit function
        @numba.njit
        def test_func(x):
            return x + 1
        test_func(1)
        ret = np.uint32(numba.get_num_threads())
    except (NameError, ImportError, TypeError):
        ret = np.uint32(0)
    if ret:
        print('numba.njit is available.', flush=True)
    else:
        print('numba.njit not available on this system.', flush=True)
    return ret


def raise_njit_not_available():
    if globals.cpu_njit_num_threads > 0:
        raise AssertionError("Invalid state encountered in raise_njit_not_available()")
    else:
        raise AssertionError("numba.njit not available on this system.")


def is_use_njit(**kwargs) -> bool:
    if globals.cpu_njit_num_threads < 1:
        return False
    val = kwargs.get('use_njit', None)
    return val is None or val

###########################################################################################



def init_cuda() -> None:
    if not globals.cuda_import_success:
        return
    globals.cuda_available = False
    
    try:
        print('Test numba.cuda drivers...', flush=True)
        minimal_grid_size = find_recommended_minimal_grid_size()
        
        # Test compatibility with np.uint64
        @numba.cuda.jit
        def test_uint64(seed: np.uint64, out: DeviceNDArray):
            idx = numba.cuda.grid(1) # type: ignore
            if idx < out.size:
                out[idx] = seed + idx

        seed_value = np.uint64(42)
        test_grid_size = minimal_grid_size
        test_block_size = 1
        test_size = test_grid_size * test_block_size
        test_data = np.empty(test_size, dtype=np.uint64) 
        d_test_data = numba.cuda.to_device(test_data)
        test_uint64[test_grid_size, test_block_size](seed_value, d_test_data) # type: ignore
        retrieved_data = d_test_data.copy_to_host()
        expected_data = seed_value + np.arange(test_size, dtype=np.uint64)
        if not np.array_equal(retrieved_data, expected_data):
            raise Exception(\
                f'CUDA version is not compatible with np.uint64.\n'
                f'Retrieved: {retrieved_data}\nExpected: {expected_data}')
        print('CUDA is available and will be used for GPU operations.', flush=True)
        globals.cuda_available = True
    except Exception as e:
        print(f'CUDA initialization failed with error:\n{e}', flush=True)

    
def print_cuda_device_attributes():
    if not globals.cuda_available:
        print("Numba or CUDA is not available. GPU operations will be disabled.")
        return
    print('Printing CUDA active device attributes:\n'+'='*50)
    context                       = numba.cuda.current_context()
    context_mem_info              = context.get_memory_info()
    device                        = context.device
    # The number of Streaming Multiprocessors (SMs) affects how many blocks can run concurrently.
    multi_processor_count         = device.MULTIPROCESSOR_COUNT
    min_grid_size                 = find_recommended_minimal_grid_size()
    max_grid_dimensions_XYZ       = (device.MAX_GRID_DIM_X,device.MAX_GRID_DIM_Y,device.MAX_GRID_DIM_Z)
    max_block_dimensions_XYZ      = (device.MAX_BLOCK_DIM_X,device.MAX_BLOCK_DIM_Y,device.MAX_BLOCK_DIM_Z)
    globals.max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    globals.warp_size             = device.WARP_SIZE
    max_shared_memory_per_block   = device.MAX_SHARED_MEMORY_PER_BLOCK
    globals.max_registers_per_block = device.MAX_REGISTERS_PER_BLOCK
    memory_bus_width_bits         = device.GLOBAL_MEMORY_BUS_WIDTH
    total_constant_memory         = device.TOTAL_CONSTANT_MEMORY
    memory_clock_rate_MHz         = device.MEMORY_CLOCK_RATE/1000 # from KHz to MHz
    print(f'    Name:                               {device.name.decode("utf-8")}')
    print(f'    Free Memory:                        {context_mem_info.free//1024} [KB]',)
    print(f'    Total Memory:                       {context_mem_info.total//1024} [KB]',)
    print(f'    Compute capability:                 {device.compute_capability[0]}.{device.compute_capability[1]}')
    print(f'    Clock rate:                         {device.CLOCK_RATE/1000:.2f} [MHz]')
    print(f'    Memory clock rate:                  {memory_clock_rate_MHz:.2f} [MHz]')
    print(f'    Memory bus width:                   {memory_bus_width_bits} bits')
    '''
    Multiply by 2: Because GDDR5 memory uses double data rate (DDR).
    Divide by 8: To convert bits to bytes.
    Divide by 1000: To convert MB/s to GB/s.
    '''
    print(f'    Memory band width (theoretical)     {2*memory_clock_rate_MHz * (memory_bus_width_bits/8) / 1000:.2f} [GByte/Sec]')
    print(f'    Number of multiprocessors:          {multi_processor_count}')
    print(f'    Minimal grid size:                  {min_grid_size}')
    print(f'    Maximum grid size:                  {max_grid_dimensions_XYZ}')
    print(f'    Maximum block dimensions:           {max_block_dimensions_XYZ}')
    print(f'    Maximum threads per block:          {globals.max_threads_per_block}')
    print(f'    Warp size:                          {globals.warp_size}')
    print(f'    Maximum shared memory per block:    {max_shared_memory_per_block} [bytes]')
    print(f'    Maximum registers per block:        {globals.max_registers_per_block}')
    print(f'    Total constant memory:              {total_constant_memory} [bytes]')
    print(f'    Asynchronous engine count:          {device.ASYNC_ENGINE_COUNT}')
    print(f'    L2 cache size:                      {device.L2_CACHE_SIZE} [bytes]')
    print(f'    ECC support enabled:                {bool(device.ECC_ENABLED)}')
    # Unavailable GPU attributes
    #print(f'    Maximum threads per multiprocessor: {device.MAX_THREADS_PER_MULTIPROCESSOR}')

'''
The number of CUDA cores per SM varies depending on the GPU architecture:

Kepler (Compute Capability 3.x): 192 CUDA cores per SM
Maxwell (Compute Capability 5.x): 128 CUDA cores per SM
Pascal (Compute Capability 6.x): 64 CUDA cores per SM
Turing (Compute Capability 7.5): 64 CUDA cores per SM
Ampere (Compute Capability 8.x): 128 CUDA cores per SM

Compute Capability	Architecture	CUDA Cores per SM
5.x	Maxwell	128
6.x	Pascal	128
7.0	Volta	64
7.5	Turing	64
8.x	Ampere	64 (FP32 cores)
'''


def find_recommended_minimal_grid_size() -> int:
    """
    Dynamically finds the minimal grid size required to avoid NumbaPerformanceWarning.
    
    Returns:
        int: The minimal grid size.
    """
    # Check if the static variable exists
    if hasattr(find_recommended_minimal_grid_size, 'save_result'):
        return find_recommended_minimal_grid_size.save_result  # Return cached value
    
    @numba.cuda.jit
    def dummy_kernel():
        pass

    grid_size = 1  # Start with 1 block

    while True:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", category=NumbaPerformanceWarning)
            try:
                dummy_kernel[grid_size, 1]()  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Kernel launch failed at grid size {grid_size}: {e}")
            # Check if any warnings were captured
            if not any(w.category == NumbaPerformanceWarning for w in captured_warnings):
                break

        grid_size += 1
        # Sanity check to avoid infinite loops
        if grid_size > 1024 * 1024:  # Arbitrary large grid size limit
            raise RuntimeError("Exceeded reasonable grid size without suppressing warnings.")

    # Cache the result using a function attribute
    find_recommended_minimal_grid_size.save_result = grid_size
    return grid_size


def raise_cuda_not_available():
    if globals.cuda_available:
        raise CudaSupportError("Invalid state encountered in raise_cuda_not_available()")
    else:
        raise AssertionError("CUDA is not available on this system.")
    

def cuda_garbage_collect() -> None:
    if not globals.cuda_available:
        raise_cuda_not_available()
    gc.collect(0) # generation 0 should be sufficient to release GPU memory
    numba.cuda.current_context().memory_manager.deallocations.clear()


def simple_data_size_to_grid_block_1D(\
        data_size: int|np.uint64|np.uint32,\
        registers_per_thread: int|None = None,\
        debug: int|None = None) -> tuple[np.uint32, np.uint32]:
    if not globals.cuda_available:
        raise_cuda_not_available()  
    if debug is None:
        debug = int(globals.grid_block_shape_debug)
    block_size = calc_block_size(data_size=data_size, registers_per_thread=registers_per_thread)
    grid_size = np.uint32((data_size + block_size - 1) // block_size)
    if debug > 0:
        print(f'simple_data_size_to_grid_block_1D({data_size=}) --> {grid_size=} {block_size=}')
    return grid_size, block_size


def calc_block_size(data_size: int|np.uint64|np.uint32,\
                    registers_per_thread: int|np.uint64|np.uint32|None = None) -> np.uint32:
    if not globals.cuda_available:
        raise_cuda_not_available()    
    assert data_size > 0
    max_threads_per_block = [globals.max_threads_per_block, data_size]
    if registers_per_thread is not None:
        assert registers_per_thread > 0
        max_threads_per_block.append(globals.max_registers_per_block // registers_per_thread)
    block_size = min(max_threads_per_block)
    if block_size > globals.warp_size:
        block_size -= block_size % globals.warp_size
    assert block_size > 0
    return np.uint32(block_size)


def is_use_gpu(**kwargs) -> bool:
    if not globals.cuda_available:
        return False
    val = kwargs.get('use_gpu', None)
    return val is None or val

############################################################

globals.cpu_njit_num_threads = init_njit()

init_cuda()

print_cuda_device_attributes()

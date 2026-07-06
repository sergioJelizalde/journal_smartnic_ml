The C API headers here (onnxruntime_c_api.h, onnxruntime_ep_c_api.h,
onnxruntime_float16.h) are vendored from the onnxruntime v1.24.4 release
tag to match `pip show onnxruntime` in this environment.

They are headers only -- no .so is vendored. To link bench_onnx_capi.c:
  x86:  point -L/-Wl,-rpath at your `python3 -c "import onnxruntime,os;
        print(os.path.dirname(onnxruntime.__file__)+'/capi')"` directory.
        That directory ships libonnxruntime.so.<version> but not the
        unversioned libonnxruntime.so symlink gcc's -lonnxruntime wants --
        create it once:
          ln -s libonnxruntime.so.<version> libonnxruntime.so
          ln -s libonnxruntime.so.<version> libonnxruntime.so.<major>
  BF3:  point at the libonnxruntime.so you already built from source on
        the DPU (per your existing ORT-C-API-on-BF3 setup).

import importlib.metadata

required_packages = [
    "wcwidth", "tokenmonster", "sentencepiece", "pytz", "namex", "mpmath", "libclang",
    "flatbuffers", "bitarray", "argparse", "antlr4-python3-runtime", "xxhash", "wrapt",
    "urllib3", "tzdata", "typing-extensions", "typing", "tqdm", "termcolor",
    "tensorflow-io-gcs-filesystem", "tensorboard-data-server", "tabulate", "sympy", "six",
    "safetensors", "rpds-py", "regex", "pyyaml", "python-dotenv", "pygments", "pyflakes",
    "pycparser", "pycodestyle", "pyasn1", "pyarrow-hotfix", "pulp", "psutil", "protobuf",
    "PrettyTable", "portalocker", "pluggy", "pillow", "packaging", "nvidia-nvtx-cu12",
    "nvidia-nvjitlink-cu12", "nvidia-nccl-cu12", "nvidia-curand-cu12", "nvidia-cufft-cu12",
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cupti-cu12",
    "nvidia-cublas-cu12", "numpy", "networkx", "mypy-extensions", "multidict", "mdurl",
    "mccabe", "MarkupSafe", "markdown", "lxml", "jmespath", "iniconfig", "idna", "grpcio",
    "gast", "fsspec", "frozenlist", "frozendict", "filelock", "einops", "docutils", "dill",
    "cython", "colorama", "click", "charset-normalizer", "certifi", "cachetools", "beartype",
    "attrs", "absl-py", "yarl", "werkzeug", "typing-inspect", "triton", "scipy", "sacrebleu",
    "rsa", "requests", "referencing", "python-dateutil", "pytest", "pyarrow", "optree",
    "opt-einsum", "omegaconf", "nvidia-cusparse-cu12", "nvidia-cudnn-cu12", "numexpr",
    "multiprocess", "ml-dtypes", "markdown-it-py", "jinja2", "h5py", "google-pasta", "flake8",
    "einx", "einops-exts", "cffi", "astunparse", "aiosignal", "time-machine", "tiktoken",
    "tensorboard", "rich", "pandas", "nvidia-cusolver-cu12", "libcst", "jsonschema-specifications",
    "jaxlib", "hydra-core", "huggingface-hub", "cryptography", "botocore", "bitsandbytes",
    "aiohttp", "torchfix", "torch", "tokenizers", "s3transfer", "pendulum", "keras", "jsonschema",
    "jax", "vector-quantize-pytorch", "transformers", "torchvision", "torchdiffeq", "torchaudio",
    "tensorflow", "local-attention", "lion-pytorch", "datasets", "boto3", "awscli", "accelerate",
    "timm", "skypilot", "fairseq", "colt5-attention", "zetascale"
]

installed_packages = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}
missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

if missing_packages:
    print("Missing packages:", missing_packages)
else:
    print("All packages are installed.")
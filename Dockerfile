FROM yaohui1998/ultralytics-jetpack5:1.0

# ติดตั้ง dependencies เพิ่มเติม
RUN pip install -q scikit-image loguru rich scipy

WORKDIR /workspace

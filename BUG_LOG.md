Known package incompatabilties
------------------------------
- `numpy==1.16.1` required to avoid error:
    ```python
    ValueError: numpy.ufunc size changed, may indicate binary incompatibility. Expected 216 from C header, got 192 from PyObject
   ```
- `pytest>=4.?.?` to avoid avoid depreciation issue with `attrs>=19.1.0`

System package requirements
---------------------------
- To avoid error in opencv-python need to install:
    ```
    apt-get install -y libsm6 libxext6 libxrender-dev
    ```

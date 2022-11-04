Zarr Visualization
========


Zarr is  a format for storing chunked, compressed, N-dimensional arrays.
Below shows different use cases of how Zarr can be visualized.



<br />

- ## Bioimaging

#### Skin Layers Dermis and Epidermis
https://webknossos.org/datasets/scalable_minds/skin

```
>>> z = zarr.open_array(
    "https://data-humerus.webknossos.org/data/zarr/scalable_minds/skin/color/4-4-1"
)
>>> print(z.shape)
>>> plt.imshow(z[:, :, :, 0].T)
(3, 320, 240, 1)
<matplotlib.image.AxesImage at 0x7f4e337be510>
```
![image1](/docs/images/image1.png)


#### Mouse Cortex Layer 4 with Segmentation
https://webknossos.org/datasets/scalable_minds/l4dense_motta_et_al_demo_v2

```
>>> z = zarr.open_array("https://data-humerus.webknossos.org/data/zarr/scalable_minds/l4dense_motta_et_al_demo_v2/color/2-2-1")
>>> print(z.shape)
>>> f, axarr = plt.subplots(1, 2)
axarr[0].imshow(z[0, 1000:1800, 500:1500, 1500].T, cmap="gray")
>>> z = zarr.open_array("https://data-humerus.webknossos.org/data/zarr/scalable_minds/l4dense_motta_et_al_demo_v2/segmentation/2-2-1")
axarr[1].imshow(z[0, 1000:1800, 500:1500, 1500].T, cmap="tab20")
(1, 2786, 4254, 3413)
<matplotlib.image.AxesImage at 0x7f4e291eef50>
```
![image2](/docs/images/image2.png)


#### Confocal imaging of mouse blastocysts
```
>>> z = zarr.open_array(
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/6001240.zarr/1"
)
>>> print(z.shape)
>>> plt.imshow(z[0, 1, 100, :, :], cmap="Oranges")
(1, 2, 236, 137, 135)
<matplotlib.image.AxesImage at 0x7f4e2b251310>
```
![image3](/docs/images/image3.png)

<br />


- ## Geo-spatial Datasets

#### NASA Prediction of Worldwide Energy Resources (POWER)
https://registry.opendata.aws/nasa-power

```
>>> g = zarr.open_group(
    "s3://power-analysis-ready-datastore/power_901_monthly_meteorology_utc.zarr",
    storage_options={"anon": True}
)
>>> print(list(g.keys()))
>>> z = g["EVLAND"]
>>> print(z.shape)
>>> plt.imshow(z[200], origin="lower", cmap="hot")
['CDD0', 'CDD10', 'CDD18_3', 'DISPH', 'EVLAND', 'EVPTRNS', 'FROST_DAYS', 'FRSEAICE', 'FRSNO', 'GWETPROF', 'GWETROOT', 'GWETTOP', 'HDD0', 'HDD10', 'HDD18_3', 'PBLTOP', 'PRECSNO', 'PRECSNOLAND', 'PRECSNOLAND_SUM', 'PRECTOTCORR', 'PRECTOTCORR_SUM', 'PS', 'QV10M', 'QV2M', 'RH2M', 'RHOA', 'SLP', 'SNODP', 'T10M', 'T10M_MAX', 'T10M_MAX_AVG', 'T10M_MIN', 'T10M_MIN_AVG', 'T10M_RANGE', 'T10M_RANGE_AVG', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MAX_AVG', 'T2M_MIN', 'T2M_MIN_AVG', 'T2M_RANGE', 'T2M_RANGE_AVG', 'TO3', 'TQV', 'TROPPB', 'TROPQ', 'TROPT', 'TS', 'TSOIL1', 'TSOIL2', 'TSOIL3', 'TSOIL4', 'TSOIL5', 'TSOIL6', 'TSURF', 'TS_MAX', 'TS_MAX_AVG', 'TS_MIN', 'TS_MIN_AVG', 'TS_RANGE', 'TS_RANGE_AVG', 'U10M', 'U2M', 'U50M', 'V10M', 'V2M', 'V50M', 'WD10M', 'WD2M', 'WD50M', 'WS10M', 'WS10M_MAX', 'WS10M_MAX_AVG', 'WS10M_MIN', 'WS10M_MIN_AVG', 'WS10M_RANGE', 'WS10M_RANGE_AVG', 'WS2M', 'WS2M_MAX', 'WS2M_MAX_AVG', 'WS2M_MIN', 'WS2M_MIN_AVG', 'WS2M_RANGE', 'WS2M_RANGE_AVG', 'WS50M', 'WS50M_MAX', 'WS50M_MAX_AVG', 'WS50M_MIN', 'WS50M_MIN_AVG', 'WS50M_RANGE', 'WS50M_RANGE_AVG', 'Z0M', 'lat', 'lon', 'time']
(492, 361, 576)
<matplotlib.image.AxesImage at 0x7f4e26b990d0>
```
![image4](/docs/images/image4.png)


#### Atmospheric Conditions from the Coupled Model Intercomparison Project Phase 6 (CMIP6)
https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.AS-RCEC.TaiESM1.1pctCO2
```
>>> g = zarr.open_group(
    "s3://cmip6-pds/CMIP6/CMIP/AS-RCEC/TaiESM1/1pctCO2/r1i1p1f1/Amon/hfls/gn/v20200225/",
    storage_options={"anon": True}
)
>>> print(list(g.keys()))
>>> z = g["hfls"]
>>> print(z.shape)
>>> plt.imshow(z[900], origin="lower")
['hfls', 'lat', 'lat_bnds', 'lon', 'lon_bnds', 'time', 'time_bnds']
(1800, 192, 288)
<matplotlib.image.AxesImage at 0x7f4e29245310>
```
![image5](/docs/images/image5.png)


#### SILAM Air Quality
https://registry.opendata.aws/silam/

```
>>> from datetime import datetime

>>> date = datetime.today().strftime('%Y%m%d')
>>> print("date", date)

>>> z = zarr.open_array(
    f"s3://fmi-opendata-silam-surface-zarr/global/{date}/silam_glob_v5_7_1_{date}_SO2_d0.zarr/SO2/",
    storage_options={"anon": True}
)
>>> print(z.shape)
>>> plt.imshow(np.log(z[20]), origin="lower", cmap="Spectral")
date 20220825
(24, 897, 1800)
<matplotlib.image.AxesImage at 0x7f4e2710f790>
```
![image6](/docs/images/image6.png)


#### Estimating the Circulation and Climate of the Ocean (ECCO)
https://catalog.pangeo.io/browse/master/ocean/ECCOv4r3/

```
>>> z = zarr.open_array(
    "https://storage.googleapis.com/pangeo-data/ECCO_basins.zarr/basin_mask"
  )
>>> print(z.shape)
>>> plt.imshow(z[10])
(13, 90, 90)
<matplotlib.image.AxesImage at 0x7f4e2b38fbd0>
```
![image7](/docs/images/image7.png)



#### Forest risks 
https://decks.carbonplan.org/pangeo-showcase/10-27-21

```
src = 'https://ncsa.osn.xsede.org/Pangeo/pangeo-forge/gcp-feedstock/gpcp.zarr'
reticulate::source_python("read_zarr.py")
x = read_zarr(src, "precip", slice = 1L)[1L, , , drop = TRUE]
xs = read_zarr_raw(src, "lon_bounds")
ys = read_zarr_raw(src, "lat_bounds")
ex = c(range(xs), range(ys))
ximage::ximage(x[nrow(x):1, ], extent = ex, col = hcl.colors(256))
maps::map("world2", add = TRUE)
```
![image8](/docs/images/image8.png)

EXTRA 1
```
import xarray as xr
import fsspec

store = fsspec.get_mapper('https://carbonplan.blob.core.windows.net/carbonplan-forests/risks/results/web/fire.zarr')

ds = xr.open_zarr(store, consolidated=True)
```
![image9](/docs/images/image9.jpg)

EXTRA 2
```
import s3fs
import xarray as xr

fs = s3fs.S3FileSystem(anon=True)
mapper = fs.get_mapper("s3://cmip6-pds/CMIP6/CMIP/AS-RCEC/TaiESM1/1pctCO2/r1i1p1f1/Amon/hfls/gn/v20200225/")
ds = xr.open_zarr(mapper, consolidated=True, decode_times=False)
print(list(ds.keys()))
z = ds["hfls"]
print(z.shape)
plt.imshow(z[200], origin="lower", cmap="hot")
```
![image10](/docs/images/image10.png)




```
store = zarr.RedisStore(port=args.port)
root = zarr.group(store=store, overwrite=True)
t = 0
while True:
    arr = root.zeros(f"{t}", shape=grid.shape, chunks=(25, 25))  # create a new array for this timestep
    arr[…] = grid # write data to zarr array
    
    t += 1  # increment the time counter
    time.sleep(update_interval)
    grid = update(grid, N)  # evolve the model one time step
```
![image11](/docs/images/image11.gif)


```
>>> import tensorstore as ts
>>> import numpy as np
>>> # Create a zarr array on the local filesystem
>>> dataset = ts.open({
...     'driver': 'zarr',
...     'kvstore': 'file:///tmp/my_dataset/',
... },
... dtype=ts.uint32,
... chunk_layout=ts.ChunkLayout(chunk_shape=[256, 256, 1]),
... create=True,
... shape=[5000, 6000, 7000]).result()
>>> # Create two numpy arrays with example data to write.
>>> a = np.arange(100*200*300, dtype=np.uint32).reshape((100, 200, 300))
>>> b = np.arange(200*300*400, dtype=np.uint32).reshape((200, 300, 400))
>>> # Initiate two asynchronous writes, to be performed concurrently.
>>> future_a = dataset[1000:1100, 2000:2200, 3000:3300].write(a)
>>> future_b = dataset[3000:3200, 4000:4300, 5000:5400].write(b)
>>> # Wait for the asynchronous writes to complete
>>> future_a.result()
>>> future_b.result()

```
![image12](/docs/images/image12.jpg)


### A layout of multiple chunked 3D array 

![image13](/docs/images/image13.gif)
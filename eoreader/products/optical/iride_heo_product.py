from __future__ import annotations

from datetime import datetime
import json
from functools import cache
from pathlib import Path
from typing import Dict

import numpy as np
import geopandas as gpd
import shapely
import pandas as pd
import xarray as xr
import rasterio
from rasterio import crs
from lxml import etree
from sertit import geometry, vectors, xml, path
from sertit.types import AnyPathStrType, AnyPathType
from sertit.misc import ListEnum

from shapely.geometry import shape

from eoreader import DATETIME_FMT, cache, utils
from eoreader.bands import (
    PAN,
    BLUE,
    GREEN,
    RED,
    VRE_1,
    VRE_2,
    VRE_3,
    NIR,
    BandNames,
    SpectralBand
)
from eoreader.exceptions import InvalidProductError
from eoreader.products.optical.optical_product import OpticalProduct
from eoreader.stac import PROJ_CODE, CENTER_WV, FWHM, GSD, ID, NAME
from eoreader.utils import simplify

try:
    from pystac import Item
except ModuleNotFoundError:
    from typing import Any as Item



class IrideHeoProduct(OpticalProduct):
    """
    IRIDE Hawk for Earth Observation OPT8
    """
    def __init__(
        self,
        product_path: AnyPathStrType,
        archive_path: AnyPathStrType = None,
        output_path: AnyPathStrType = None,
        remove_tmp: bool = False,
        **kwargs,
    ) -> None:
                # Initialization from the super class
        super().__init__(product_path, archive_path, output_path, remove_tmp, **kwargs)


    def _pre_init(self, **kwargs) -> None:
        """
        Function used to pre_init the products
        """
        self._has_cloud_cover = False
        self.needs_extraction = False

        # Get STAC Item
        self.item = self._set_item()

        # Pre init done by the super class
        super()._pre_init(**kwargs)

    @classmethod
    def _is_product(cls, product_path: Path) -> bool:
        name = product_path.stem if product_path.suffix == ".zip" else product_path.name
        return name.startswith("IMH01_") and "OPT8" in name
    
    @cache
    def _set_item(self) -> Item:
        """
        Read STAC Item metadata (JSON) and output it as a metadata XML root
        and an empty namespaces dict, following EOReader GeoJSON conventions.
        """
        mtd_from_path = "*.json"
        mtd_archived = r".*\.json"
        
        try:
            import pystac

            if self.is_archived:
                raw = self._read_archived_file(mtd_archived)
                return pystac.Item.from_dict(json.loads(raw))
            else:
                stac_path = next(self.path.glob(mtd_from_path))
                return pystac.Item.from_file(stac_path)
            
        except ModuleNotFoundError as exc:
            raise InvalidProductError(
                "You should install 'pystac' to use STAC Products."
            ) from exc
        
        except Exception as exc:
            raise InvalidProductError(
                f"Invalid metadata JSON for {self.path}!"
            ) from exc
  
    
    @cache
    def _read_mtd(self) -> (etree._Element, dict):
        """
        Read STAC Item metadata (JSON) and output it as a metadata XML root
        and an empty namespaces dict, following EOReader GeoJSON conventions.
        """
        mtd_from_path = "**/*.json"
        mtd_archived = r".*\.json"
        
        try:
            if self.is_archived:
                # Read STAC JSON as a vector-like structure
                gdf = self._read_archived_vector(archive_regex=mtd_archived)

            else:
                try:
                    mtd_file = self.path / mtd_from_path
                    if not mtd_file.exists():
                        raise StopIteration
                    gdf = vectors.read(mtd_file)
                except StopIteration as ex:
                    raise InvalidProductError(
                        f"Metadata file ({mtd_from_path}) not found in {self.path}"
                    ) from ex

        except etree.XMLSyntaxError as exc:
            raise InvalidProductError(
                f"Invalid metadata JSON for {self.path}!"
            ) from exc


        if "datetime" in gdf.columns:
            gdf["datetime"] = pd.to_datetime(gdf["datetime"]).dt.strftime(DATETIME_FMT)

        if "start_datetime" in gdf.columns:
            gdf["start_datetime"] = pd.to_datetime(
                gdf["start_datetime"]
            ).dt.strftime(DATETIME_FMT)

        if "end_datetime" in gdf.columns:
            gdf["end_datetime"] = pd.to_datetime(
                gdf["end_datetime"]
            ).dt.strftime(DATETIME_FMT)

        root = xml.df_to_xml(gdf)

        return root, {}


    @cache
    def extent(self) -> gpd.GeoDataFrame:
        """
        Get UTM extent of stack.

        Returns:
            gpd.GeoDataFrame: Extent in UTM
        """
        extent_geom = geometry.from_bounds_to_polygon(*self.item.bbox)
        if not isinstance(extent_geom, list):
            extent_geom = [extent_geom]
        # Get extent
        return gpd.GeoDataFrame(
            geometry=extent_geom,
            crs=vectors.EPSG_4326,
        ).to_crs(self.crs())

    @cache
    @simplify
    def footprint(self) -> gpd.GeoDataFrame:
        """
        Get UTM footprint of the products (without nodata, *in french == emprise utile*)

        .. code-block:: python

            >>> from eoreader.reader import Reader
            >>> path = r"S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432.SAFE.zip"
            >>> prod = Reader().open(path)
            >>> prod.footprint()
               index                                           geometry
            0      0  POLYGON ((199980.000 4500000.000, 199980.000 4...

        Returns:
            gpd.GeoDataFrame: Footprint as a GeoDataFrame
        """
        # Get footprint
        try:
            footprint = gpd.GeoDataFrame.from_dict(
                data=shapely.polygons(self.item.geometry["coordinates"]),
                crs=vectors.EPSG_4326,
            )
        except ValueError:
            footprint = gpd.GeoDataFrame(
                geometry=shapely.polygons(self.item.geometry["coordinates"]),
                crs=vectors.EPSG_4326,
            )
        return footprint.to_crs(self.crs())

    @cache
    def crs(self) -> crs.CRS:
        """
        Get UTM projection of stack.

        Returns:
            crs.CRS: CRS object
        """
        epsg = self.item.properties.get(PROJ_CODE)

        if epsg is None:
            def_crs = gpd.GeoDataFrame(
                geometry=geometry.from_bounds_to_polygon(*self.item.bbox),
                crs=vectors.EPSG_4326,
            ).estimate_utm_crs()
        else:
            try:
                def_crs = crs.CRS.from_epsg(code=epsg)
            except ValueError:
                def_crs = crs.CRS.from_string(epsg)

        return def_crs


    def _get_name_constellation_specific(self) -> str:
        return self.item.id
    
    
    def _get_condensed_name(self) -> str:
        """
        Get IRIDE HEO products condensed name ({date}_{constellation}_{product_type}_{unique_id}).

        Returns:
            str: Condensed name
        """
        platform = self.item.properties.get("platform").replace("-", "_")
        return f"{self.get_datetime()}_{platform}_{self.product_type}"


    def get_datetime(self, as_datetime: bool = False) -> str | datetime:
        dt_str = self.item.properties.get("datetime")

        if not dt_str:
            raise InvalidProductError("STAC item has no 'datetime' property")

        product_datetime = datetime.fromisoformat(
            dt_str.replace("Z", "+00:00")
        )

        return product_datetime if as_datetime else product_datetime.strftime(DATETIME_FMT)


    def _set_instrument(self) -> None:
        """
        Set instrument
        """
        self.instrument = "MSI"


    def _set_product_type(self) -> None:
        """Set products type"""
        product_type = self.split_name[1]

        if product_type.startswith("0"):
            self.product_type = "0ST"
        else:
            self.product_type = product_type


    def _set_pixel_size(self) -> None:
        """
        Set product default pixel size (in meters)
        Extracted from the GSD of the BLUE band
        """
        for asset in self.item.assets.values():
            bands = asset.extra_fields.get("bands")
            if not bands:
                continue

            for band in bands:
                if band.get("eo:common_name") == "blue":
                    gsd = band.get("gsd")
                    if gsd is not None:
                        self.pixel_size = float(gsd)
                        return

        raise InvalidProductError(
            "Unable to determine pixel size from BLUE band GSD in STAC assets"
        )


    def _map_bands(self) -> Dict[BandNames, SpectralBand]:
        band_map = {
            "pan": PAN,
            "blue": BLUE,
            "green": GREEN,
            "red01": RED,
            "red02": VRE_1,
            "red03": VRE_2,
            "red04": VRE_3,
            "nir": NIR,
        }

        bands: Dict[BandNames, SpectralBand] = {}

        for asset_key, band_name in band_map.items():
            asset = self.item.assets.get(asset_key)
            if asset is None:
                continue

            # STAC Raster extension: bands live in asset.extra_fields["bands"]
            asset_bands = asset.extra_fields.get("bands")
            if not asset_bands:
                continue

            band_info = asset_bands[0]

            bands[band_name] = SpectralBand(
                eoreader_name=band_name,
                **{NAME: band_info.get("name"),
                   ID: band_info.get("name"),
                   GSD: band_info.get("gsd"),
                   CENTER_WV: (
                        band_info.get("eo:center_wavelength") * 1000
                        if band_info.get("eo:center_wavelength") is not None
                        else None
                   ),
                   FWHM: band_info.get("eo:full_width_half_max")
                },
            )

        return self.bands.map_bands(bands)


    def get_band_paths(
        self, band_list: list, pixel_size: float = None, **kwargs
    ) -> dict:
        """
        Return the paths of required bands.

        .. code-block:: python

            >>> from eoreader.reader import Reader
            >>> from eoreader.bands import *
            >>> path = r"SENTINEL2A_20190625-105728-756_L2A_T31UEQ_C_V2-2"
            >>> prod = Reader().open(path)
            >>> prod.get_band_paths([GREEN, RED])
            {
                <SpectralBandNames.GREEN: 'GREEN'>:
                'SENTINEL2A_20190625-105728-756_L2A_T31UEQ_C_V2-2/SENTINEL2A_20190625-105728-756_L2A_T31UEQ_C_V2-2_FRE_B3.tif',
                <SpectralBandNames.RED: 'RED'>:
                'SENTINEL2A_20190625-105728-756_L2A_T31UEQ_C_V2-2/SENTINEL2A_20190625-105728-756_L2A_T31UEQ_C_V2-2_FRE_B4.tif'
            }

        Args:
            band_list (list): List of the wanted bands
            pixel_size (float): Band pixel size
            kwargs: Other arguments used to load bands

        Returns:
            dict: Dictionary containing the path of each queried band
        """
        band_paths = {}
        for band in band_list:
            # Get clean band path
            clean_band = self.get_band_path(band, pixel_size=pixel_size, **kwargs)
            if clean_band.is_file():
                band_paths[band] = clean_band
            else:
                band_id = f"{self.bands[band].id}"

                try:
                    if self.is_archived:
                        band_paths[band] = self._get_archived_rio_path(
                            f".*{band_id}.tif",
                        )
                    else:
                        band_paths[band] = path.get_file_in_dir(
                            f"{self.path}/IMG_DATA",
                            pattern_str=f"{band_id}.tif"
                        )
                except (FileNotFoundError, IndexError) as ex:
                    raise InvalidProductError(
                        f"Non existing {band} ({band_id}) band for {self.path}"
                    ) from ex

        return band_paths
    

    def _read_band(
        self,
        band_path: AnyPathType,
        band: BandNames = None,
        pixel_size: tuple | list | float = None,
        size: list | tuple = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Read band from disk.

        .. WARNING::
            Invalid pixels are not managed here

        Args:
            band_path (AnyPathType): Band path
            band (BandNames): Band to read
            pixel_size (tuple | list | float): Size of the pixels of the wanted band, in dataset unit (X, Y)
            size (tuple | list): Size of the array (width, height). Not used if pixel_size is provided.
            kwargs: Other arguments used to load bands
        Returns:
            xr.DataArray: Band xarray

        """
        resampling = kwargs.pop("resampling", self.band_resampling)
        with rasterio.open(str(band_path)) as dst:
            # Manage the case if we open a simple band (EOReader processed bands)
            if dst.count == 1:
                # Read band
                band_arr = utils.read(
                    band_path,
                    pixel_size=pixel_size,
                    size=size,
                    resampling=resampling,
                    **kwargs,
                )

            # Manage the case if we open a stack (native DIMAP bands)
            else:
                band_arr = utils.read(
                    band_path,
                    pixel_size=pixel_size,
                    size=size,
                    resampling=resampling,
                    indexes=[self.bands[band].id],
                    **kwargs,
                )

        # Pop useless long name
        if "long_name" in band_arr.attrs:
            band_arr.attrs.pop("long_name")

        # To float32
        # if band_arr.dtype != np.float32:
        #     band_arr = band_arr.astype(np.float32)

        return band_arr


    def _to_reflectance(
        self,
        band_arr: xr.DataArray,
        band_path: AnyPathType,
        band: BandNames,
        **kwargs,
    ) -> xr.DataArray:
        """
        Converts band to reflectance

        Args:
            band_arr (xr.DataArray): Band array to convert
            band_path (AnyPathType): Band path
            band (BandNames): Band to read
            **kwargs: Other keywords

        Returns:
            xr.DataArray: Band in reflectance
        """
        return band_arr


    def get_cloud_cover(self) -> float | None:
        return None
    
    
    def get_mean_sun_angles(self) -> (float, float):
        """
        Get Mean Sun angles (Azimuth and Zenith angles)

        .. code-block:: python

            >>> from eoreader.reader import Reader
            >>> path = r"S2A_MSIL1C_20200824T110631_N0209_R137_T30TTK_20200824T150432.SAFE.zip"
            >>> prod = Reader().open(path)
            >>> prod.get_mean_sun_angles()
            (149.148155074489, 32.6627897525474)

        Returns:
            (float, float): Mean Azimuth and Zenith angle
        """
        azimuth = self.item.properties.get("view:sun_azimuth")
        elevation = self.item.properties.get("view:sun_elevation")

        if azimuth is None or elevation is None:
            raise InvalidProductError("Sun angles not found in STAC properties")

        # Zenith = 90° - elevation
        zenith = 90.0 - elevation

        return float(azimuth), float(zenith)


    def _get_quicklook_path(self) -> Path | None:
        asset = self.item["assets"].get("quicklook")
        if asset:
            return self.path / asset["href"]
        return None


    def _get_thumbnail_path(self) -> Path | None:
        asset = self.item["assets"].get("thumbnail")
        if asset:
            return self.path / asset["href"]
        return None

    
    def _load_bands(
        self,
        bands: list,
        pixel_size: float = None,
        size: list | tuple = None,
        **kwargs,
    ) -> dict:
        """
        Load bands as numpy arrays with the same pixel size (and same metadata).

        Args:
            bands (list): List of the wanted bands
            pixel_size (float): Band pixel size in meters
            size (tuple | list): Size of the array (width, height). Not used if pixel_size is provided.
            kwargs: Other arguments used to load bands
        Returns:
            dict: Dictionary {band_name, band_xarray}
        """
        # Return empty if no band is specified
        if not bands:
            return {}

        band_paths = self.get_band_paths(bands, pixel_size=pixel_size, **kwargs)

        # Open bands and get array (resampled if needed)
        band_arrays = self._open_bands(
            band_paths, pixel_size=pixel_size, size=size, **kwargs
        )

        return band_arrays


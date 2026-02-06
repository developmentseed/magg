"""
ICESat-2 ATL06 granule reader.

Implements [`GranuleReader`][magg.processing.GranuleReader] for ATL06
land-ice elevation data stored as HDF5 on S3.
"""

from __future__ import annotations

import logging

import h5coro
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Ground tracks present in every ATL06 granule.
GROUND_TRACKS: list[str] = ["gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"]


class ATL06Reader:
    """Read ATL06 land-ice elevation data from S3-hosted HDF5 granules.

    This reader knows the internal structure of ATL06 files:

    - Six ground tracks (``gt1l`` ... ``gt3r``), each an independent
      observation group.
    - Coordinates at ``/{track}/land_ice_segments/latitude`` and
      ``/{track}/land_ice_segments/longitude``.
    - Elevation at ``/{track}/land_ice_segments/h_li`` with uncertainty
      ``h_li_sigma`` and quality flag ``atl06_quality_summary``.

    It uses `h5coro <https://github.com/ICESat2-SlideRule/h5coro>`_ for
    S3 byte-range reads --- only the datasets and row ranges needed are
    fetched, avoiding full-file downloads.

    Parameters
    ----------
    s3_credentials : dict
        Temporary S3 credentials from
        [`get_nsidc_s3_credentials`][magg.auth.get_nsidc_s3_credentials].
        Accepts both camelCase (``accessKeyId``) and snake_case
        (``aws_access_key_id``) key names.
    h5coro_driver : class, optional
        h5coro driver class.  Defaults to ``h5coro.s3driver.S3Driver``.

    Examples
    --------
    ```python
    from magg.atl06 import ATL06Reader
    from magg.auth import get_nsidc_s3_credentials

    reader = ATL06Reader(get_nsidc_s3_credentials())
    groups = reader.read_coordinates("s3://nsidc-cumulus.../ATL06_...h5")
    # groups is a list of 6 (lats, lons) tuples, one per ground track
    ```
    """

    def __init__(
        self,
        s3_credentials: dict,
        h5coro_driver: type | None = None,
    ) -> None:
        if h5coro_driver is None:
            from h5coro import s3driver

            h5coro_driver = s3driver.S3Driver

        self._driver = h5coro_driver
        self._credentials = {
            "aws_access_key_id": (
                s3_credentials.get("accessKeyId") or s3_credentials.get("aws_access_key_id")
            ),
            "aws_secret_access_key": (
                s3_credentials.get("secretAccessKey") or s3_credentials.get("aws_secret_access_key")
            ),
            "aws_session_token": (
                s3_credentials.get("sessionToken") or s3_credentials.get("aws_session_token")
            ),
        }
        self._h5obj: h5coro.H5Coro | None = None

    def _open(self, granule_url: str) -> h5coro.H5Coro:
        """Open an HDF5 granule via h5coro, caching across read phases."""
        resource_path = granule_url.replace("s3://", "")
        self._h5obj = h5coro.H5Coro(
            resource_path,
            self._driver,
            credentials=self._credentials,
            errorChecking=True,
            verbose=False,
        )
        return self._h5obj

    def read_coordinates(
        self,
        granule_url: str,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Read lat/lon for all six ground tracks.

        Parameters
        ----------
        granule_url : str
            S3 URL to an ATL06 HDF5 file.

        Returns
        -------
        list of (lats, lons) tuples
            Six pairs, one per ground track in
            [`GROUND_TRACKS`][magg.atl06.GROUND_TRACKS] order.
            Tracks with no data return empty arrays.
        """
        h5obj = self._open(granule_url)
        result: list[tuple[np.ndarray, np.ndarray]] = []

        for track in GROUND_TRACKS:
            try:
                coord_data = h5obj.readDatasets(
                    [
                        f"/{track}/land_ice_segments/latitude",
                        f"/{track}/land_ice_segments/longitude",
                    ]
                )
                lats = coord_data[f"/{track}/land_ice_segments/latitude"]
                lons = coord_data[f"/{track}/land_ice_segments/longitude"]
                result.append((lats, lons))
            except Exception:
                result.append((np.array([]), np.array([])))

        return result

    def read_data(
        self,
        granule_url: str,
        group_index: int,
        row_slice: slice,
        morton_indices: np.ndarray,
    ) -> pd.DataFrame | None:
        """Read elevation, uncertainty, and quality for one ground track.

        Applies ATL06 quality filtering: only rows where
        ``atl06_quality_summary == 0`` are retained.

        Parameters
        ----------
        granule_url : str
            Same URL passed to
            [`read_coordinates`][magg.atl06.ATL06Reader.read_coordinates].
        group_index : int
            Ground track index (0--5), corresponding to position in
            [`GROUND_TRACKS`][magg.atl06.GROUND_TRACKS].
        row_slice : slice
            Bounding row range for hyperslice read.
        morton_indices : np.ndarray
            Pre-computed order-18 morton indices for the rows in
            ``row_slice``, already spatially masked to the parent cell.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ``h_li``, ``s_li``, ``midx``, or
            ``None`` if no rows pass quality filtering.
        """
        if self._h5obj is None:
            raise RuntimeError("read_coordinates must be called before read_data")

        track = GROUND_TRACKS[group_index]
        min_idx, max_idx = row_slice.start, row_slice.stop

        data = self._h5obj.readDatasets(
            [
                {
                    "dataset": f"/{track}/land_ice_segments/h_li",
                    "hyperslice": [(min_idx, max_idx)],
                },
                {
                    "dataset": f"/{track}/land_ice_segments/h_li_sigma",
                    "hyperslice": [(min_idx, max_idx)],
                },
                {
                    "dataset": f"/{track}/land_ice_segments/atl06_quality_summary",
                    "hyperslice": [(min_idx, max_idx)],
                },
            ]
        )

        h_li = data[f"/{track}/land_ice_segments/h_li"]
        s_li = data[f"/{track}/land_ice_segments/h_li_sigma"]
        q_flag = data[f"/{track}/land_ice_segments/atl06_quality_summary"]

        quality_mask = q_flag == 0
        if not np.any(quality_mask):
            return None

        return pd.DataFrame(
            {
                "h_li": h_li[quality_mask],
                "s_li": s_li[quality_mask],
                "midx": morton_indices[quality_mask],
            }
        )

# *****************************************************************************************************
# Imports
# *****************************************************************************************************
from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, ListedColormap
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

import json
import configparser
import logging


# *****************************************************************************************************
# Abstract Classes
# *****************************************************************************************************


class ImageArray(ABC):

    def __init__(self, array):
        self.image_array = array

    def get_image_array(self):
        return self.image_array

    def _save_image_array(self, save_dir, root_file_name):
        save_path = os.path.join(
            save_dir, self._get_image_array_save_name(root_file_name)
        )
        np.save(save_path, self.image_array)

    def _get_image_array_save_name(self, root_file_name):
        return root_file_name + "_" + self.image_array_suffix + ".npy"

    @abstractmethod
    def save_image_array(self):
        pass

    @abstractmethod
    def get_image_array_save_name(self):
        pass

    @property
    @abstractmethod
    def image_array_suffix(self):
        pass


class Meta(ABC):
    def __init__(self, metadata):
        self.metadata = metadata if metadata is not None else {}

    def create_meta_key(self, key, value):
        if key not in self.metadata:
            self.metadata[key] = value
        else:
            raise Exception(f"Key '{key}' already exists.")

    def update_meta_key(self, key, value):
        if key in self.metadata:
            self.metadata[key] = value
        else:
            raise Exception(f"Key '{key}' not found.")

    def get_meta_key(self, key):
        if key in self.metadata:
            return self.metadata.get(key)
        else:
            raise Exception(f"Key '{key}' not found.")

    def get_meta(self):
        return self.metadata

    def _save_meta(self, save_dir, root_file_name):
        save_path = os.path.join(save_dir, self.get_meta_save_name(root_file_name))
        with open(save_path, "w") as f:
            json.dump(self.get_meta(), f)

    def get_meta_save_name(self, root_file_name):
        return root_file_name + "_" + self.meta_suffix + ".json"

    @abstractmethod
    def save_meta(self):
        pass

    @property
    @abstractmethod
    def meta_suffix(self):
        pass


class Image(ImageArray, Meta):
    def __init__(self, array, metadata):
        self.image_array = array
        self.meta = metadata

    @property
    def image_array_suffix(self):
        return self.image_suffix + "_array"

    @property
    def meta_suffix(self):
        return self.image_suffix + "_meta"

    @abstractmethod
    def _prepare_display(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def save_image(self):
        pass

    @abstractmethod
    def get_image_save_name(self):
        pass

    def save_all(self):
        self.save_meta()
        self.save_image_array()
        self.save_image()

    @property
    @abstractmethod
    def image_suffix(self):
        pass

    @property
    def root_file_name(self):
        pass


class Transformer(ABC):
    @abstractmethod
    def apply(self, image):
        pass


# *****************************************************************************************************
# Concrete Classes
# *****************************************************************************************************

# Config
# **********************************************


class GlobalConfig(Meta):
    _instance = None

    def __new__(cls, metadata=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metadata = (
                metadata if metadata is not None else {}
            )  # Initialize metadata
            cls._instance._initialize_logging()
        return cls._instance

    def __init__(self, metadata=None):
        pass

    @property
    def meta_suffix(self):
        return "global_config"

    @property
    def SAVE_DIR(self):
        return self.get_meta_key("save_dir")

    @property
    def ROOT_DIR(self):
        return self.get_meta_key("root_dir")

    @property
    def LOG_LEVEL(self):
        return self.get_meta_key("log_level").upper()

    @property
    def META_SAVE_DIR(self):
        return self.get_meta_key("meta_save_dir").upper()

    @property
    def ARRAY_SAVE_DIR(self):
        return self.get_meta_key("array_save_dir").upper()

    @property
    def IMAGE_SAVE_DIR(self):
        return self.get_meta_key("image_save_dir").upper()

    def _initialize_logging(self):

        self.logger = logging.getLogger("RED_SKY")
        log_level = getattr(logging, self.LOG_LEVEL, logging.WARNING)
        self.logger.setLevel(log_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.debug(f"Logging initialized with level: {self.LOG_LEVEL}")

    def save_meta(self):
        return self._save_meta(self.get_meta_key("save_dir"), "")


class GlobalConfigLoader(Transformer):
    def __init__(self):
        self.metadata = {}
        self.working_dir = os.getcwd()
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read("config.ini")

    def apply(self):

        self.metadata["root_dir"] = self.config_parser.get(
            "REQUIRED", "root_dir", fallback=self.working_dir
        )

        self.metadata["save_dir"] = self.config_parser.get(
            "REQUIRED", "save_dir", fallback=self.working_dir
        )

        self.metadata["log_level"] = self.config_parser.get(
            "REQUIRED", "log_level", fallback="ERROR"
        )

        self.metadata["meta_save_dir"] = self.config_parser.get(
            "OPTIONAL", "meta_save_dir", fallback=self.metadata["save_dir"]
        )

        self.metadata["array_save_dir"] = self.config_parser.get(
            "OPTIONAL", "array_save_dir", fallback=self.metadata["save_dir"]
        )

        self.metadata["image_save_dir"] = self.config_parser.get(
            "OPTIONAL", "image_save_dir", fallback=self.metadata["save_dir"]
        )

        return GlobalConfig(self.metadata)


# RGBNIR
# **********************************************


class RGBNIRArrayLoader(Transformer):
    def __init__(
        self,
        sample_factor=None,
        normalize=True,
        band_map={"Red": 1, "Green": 2, "Blue": 3, "NIR": 4},
    ):
        self.sample_factor = sample_factor
        self.normalize = normalize
        self.band_map = band_map

    def sample_image(self, band):
        return band[:: self.sample_factor, :: self.sample_factor]

    def normalize_band(self, band):
        if band.max() > 1.0:
            if band.max() <= 255:  # 8bit
                return band / 255.0
            elif band.max() <= 65535:  # 16bit
                return band / 65535.0
        return band

    def apply(self, source):
        self.red_band = source.read(self.band_map["Red"]).astype(float)
        self.green_band = source.read(self.band_map["Green"]).astype(float)
        self.blue_band = source.read(self.band_map["Blue"]).astype(float)
        self.nir_band = source.read(self.band_map["NIR"]).astype(float)

        if self.sample_factor is not None:
            self.red_band = self.sample_image(self.red_band)
            self.green_band = self.sample_image(self.green_band)
            self.blue_band = self.sample_image(self.blue_band)
            self.nir_band = self.sample_image(self.nir_band)

        if self.normalize:
            self.red_band = self.normalize_band(self.red_band)
            self.green_band = self.normalize_band(self.green_band)
            self.blue_band = self.normalize_band(self.blue_band)
            self.nir_band = self.normalize_band(self.nir_band)

        # reformat to array of format (height, width, channels)
        rgbnir_array = np.transpose(
            [self.red_band, self.green_band, self.blue_band, self.nir_band], (1, 2, 0)
        )

        return rgbnir_array


class RGBNIRMetaLoader(Transformer):
    def __init__(self, band_map={"Red": 1, "Green": 2, "Blue": 3, "NIR": 4}):
        self.metadata = {}
        self.band_map = band_map

    def apply(self, source):
        self.metadata["file_path"] = source.name
        self.metadata["crs"] = source.crs.to_string()
        self.metadata["nbr_of_bands"] = source.count
        self.metadata["dtype"] = source.meta["dtype"]
        self.metadata["width"] = source.meta["width"]
        self.metadata["width"] = source.meta["height"]
        self.metadata["bounding_box"] = source.bounds
        self.metadata["pixel_resolution_x"] = abs(source.transform.a)
        self.metadata["pixel_resolution_y"] = abs(source.transform.e)
        self.metadata["band_map"] = self.band_map

        return self.metadata


class RGBNIRLoader(Transformer):
    def __init__(
        self,
        sample_factor=None,
        normalize=True,
        band_map={"Red": 1, "Green": 2, "Blue": 3, "NIR": 4},
    ):
        self.array_loader = RGBNIRArrayLoader(sample_factor, normalize, band_map)
        self.meta_loader = RGBNIRMetaLoader(band_map)

    def apply(self, source, root_file_name):
        array = self.array_loader.apply(source)
        metadata = self.meta_loader.apply(source)
        return RGBNIRImage(array, metadata, root_file_name)


class RGBNIRImage(Image):

    def __init__(self, array, metadata, root_file_name):
        self.global_config = GlobalConfig()
        Meta.__init__(self, metadata)
        ImageArray.__init__(self, array)
        self._root_file_name = root_file_name

    def _prepare_display(self, annotate, save_name=None):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(self.image_array[:, :, i], cmap="gray")
            ax.axis("off")

            if annotate:
                ax.set_title(f"RGBNIR Image - Band {i+1}")

        if annotate and save_name is not None:
            fig.text(0.5, 0.001, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout()
        return plt

    def display(self, annotate=True):
        if annotate:
            save_name = self.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_display(annotate, save_name=save_name)
        display.show()

    def save_image(self, annotate=True, dpi=300):
        display = self._prepare_display(annotate, self.get_image_save_name())
        save_path = os.path.join(
            self.global_config.IMAGE_SAVE_DIR, self.get_image_save_name()
        )
        display.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        display.close()

    def get_image_save_name(self):
        return self.root_file_name + "_" + self.image_suffix + ".png"

    def save_image_array(self):
        return self._save_image_array(
            self.global_config.ARRAY_SAVE_DIR, self.root_file_name
        )

    def get_image_array_save_name(self):
        return self._get_image_array_save_name(self.root_file_name)

    def save_meta(self):
        return self._save_meta(self.global_config.META_SAVE_DIR, self.root_file_name)

    @property
    def image_suffix(self):
        return "RGBNIR_image"

    @property
    def root_file_name(self):
        return self._root_file_name


# RGB
# **********************************************


class RGBArrayTransformer(Transformer):
    def __init__(
        self,
        band_map={"Red": 1, "Green": 2, "Blue": 3},
    ):
        self.band_map = band_map

    def apply(self, image_array):
        self.red_band = image_array[:, :, self.band_map["Red"] - 1]
        self.green_band = image_array[:, :, self.band_map["Green"] - 1]
        self.blue_band = image_array[:, :, self.band_map["Blue"] - 1]

        rgb_array = np.stack((self.red_band, self.green_band, self.blue_band), axis=-1)

        return rgb_array


class RGBMetaTransformer(Transformer):
    def __init__(self, band_map={"Red": 1, "Green": 2, "Blue": 3}):
        self.band_map = band_map

    def apply(self, metadata):
        self.metadata = metadata
        self.metadata["nbr_of_bands"] = 3
        self.metadata["band_map"] = self.band_map

        return self.metadata


class RGBTransformer(Transformer):
    def __init__(
        self,
        band_map={"Red": 1, "Green": 2, "Blue": 3},
    ):
        self.array_transfomer = RGBArrayTransformer(band_map)
        self.meta_transfomer = RGBMetaTransformer(band_map)

    def apply(self, rgbnir_image, root_file_name):
        image_array = rgbnir_image.get_image_array()
        metadata = rgbnir_image.get_meta()
        array = self.array_transfomer.apply(image_array)
        metadata = self.meta_transfomer.apply(metadata)
        return RGBImage(array, metadata, root_file_name)


class RGBImage(Image):

    def __init__(self, array, metadata, root_file_name):
        self.global_config = GlobalConfig()
        Meta.__init__(self, metadata)
        ImageArray.__init__(self, array)
        self._root_file_name = root_file_name

    def _prepare_display(self, annotate, save_name=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.image_array[:, :, :3])
        ax.axis("off")

        if annotate:
            ax.set_title("RGB Image")

            if save_name is not None:
                fig.text(0.5, 0.001, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout()
        return plt

    def display(self, annotate=True):
        if annotate:
            save_name = self.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_display(annotate, save_name=save_name)
        display.show()

    def save_image(self, annotate=True, dpi=300):
        display = self._prepare_display(annotate, self.get_image_save_name())
        save_path = os.path.join(
            self.global_config.IMAGE_SAVE_DIR, self.get_image_save_name()
        )
        display.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        display.close()

    def get_image_save_name(self):
        return self.root_file_name + "_" + self.image_suffix + ".png"

    def save_image_array(self):
        return self._save_image_array(
            self.global_config.ARRAY_SAVE_DIR, self.root_file_name
        )

    def get_image_array_save_name(self):
        return self._get_image_array_save_name(self.root_file_name)

    def save_meta(self):
        return self._save_meta(self.global_config.META_SAVE_DIR, self.root_file_name)

    @property
    def image_suffix(self):
        return "RGB_image"

    @property
    def root_file_name(self):
        return self._root_file_name


# Continuous NDVI
# **********************************************


class NDVIArrayTransformer(Transformer):
    def __init__(
        self,
        band_map={"Red": 1, "Green": 2, "Blue": 3, "NIR": 4},
    ):
        self.band_map = band_map

    def _calculate_ndvi(self, red_band, nir_band):
        ndvi = np.true_divide((nir_band - red_band), (nir_band + red_band))
        ndvi = np.nan_to_num(ndvi, nan=0)
        return ndvi

    def _stretch_ndvi(self, ndvi_array):
        ndvi_min = np.round(np.min(ndvi_array), 2)
        ndvi_max = np.round(np.max(ndvi_array), 2)
        ndvi_stretched = 2 * (ndvi_array - ndvi_min) / (ndvi_max - ndvi_min) - 1
        return ndvi_stretched

    def apply(self, image_array, stretch=True):
        self.red_band = image_array[:, :, self.band_map["Red"] - 1]
        self.nir_band = image_array[:, :, self.band_map["NIR"] - 1]

        ndvi_array = self._calculate_ndvi(self.red_band, self.nir_band)

        if stretch:
            ndvi_array = self._stretch_ndvi(ndvi_array)

        ndvi_array = np.clip(ndvi_array, -0.9999, 0.9999)

        return ndvi_array


class NDVIMetaTransformer(Transformer):
    def __init__(self):
        pass

    def apply(self, metadata, cmap="RdBu_r"):
        self.metadata = metadata
        self.metadata["nbr_of_bands"] = 1
        self.metadata["band_map"] = {"NDVI": 1}
        self.metadata["cmap"] = cmap
        return self.metadata


class NDVITransformer(Transformer):
    def __init__(
        self, band_map={"Red": 1, "Green": 2, "Blue": 3, "NIR": 4}, stretch=True
    ):
        self.array_transfomer = NDVIArrayTransformer(band_map)
        self.meta_transfomer = NDVIMetaTransformer()
        self.stretch = stretch

    def apply(self, rgbnir_image, root_file_name, cmap="RdBu_r"):
        image_array = rgbnir_image.get_image_array()
        metadata = rgbnir_image.get_meta()
        array = self.array_transfomer.apply(image_array, self.stretch)
        metadata = self.meta_transfomer.apply(metadata)
        return NDVIImage(array, metadata, root_file_name)


class NDVIAnalyzer:
    def __init__(self, image):
        self.ndvi_image = image

    def _prepare_NDVI_hist(self, annotate, save_name=None, bins=100):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.yaxis.set_visible(False)
        plt.xlabel("NDVI Values")
        norm = plt.Normalize(0, bins - 1)
        color_values = np.arange(bins)
        cmap = plt.get_cmap(self.ndvi_image.cmap)
        hex_colors = [to_hex(cmap(norm(value))) for value in color_values]

        _, bins, patches = plt.hist(
            self.ndvi_image.get_image_array().flatten(),
            bins=len(hex_colors),
            edgecolor="white",
            alpha=1.0,
        )

        # adapt the color of each patch
        for c, p in zip(hex_colors, patches):
            p.set_facecolor(c)

        if annotate:
            ax.set_title("NDVI Histogram")
            if save_name is not None:
                fig.text(0.5, 0.001, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout(pad=2)
        return plt

    def display_NDVI_hist(self, annotate=True, bins=100):
        if annotate:
            save_name = self.ndvi_image.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_NDVI_hist(annotate, save_name=save_name, bins=bins)
        display.show()


class NDVIImage(Image):
    def __init__(self, array, metadata, root_file_name, cmap="RdBu_r"):
        Meta.__init__(self, metadata)
        ImageArray.__init__(self, array)
        self.global_config = GlobalConfig()
        self._analyzer = NDVIAnalyzer(self)  # Composition: Image contains an Analyzer
        self.save_dir = self.global_config.SAVE_DIR
        self._root_file_name = root_file_name
        self._cmap = cmap

    def _prepare_display(self, annotate, save_name=None):
        fig, ax = plt.subplots(figsize=(8, 10))
        cax = ax.imshow(self.image_array, cmap=self.cmap, vmin=-1, vmax=1)
        cbar = fig.colorbar(cax, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
        cbar.set_label("")
        ax.axis("off")

        if annotate:
            ax.set_title("NDVI Image")
            if save_name is not None:
                fig.text(0.5, 0.09, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout()
        return plt

    def display(self, annotate=True):
        if annotate:
            save_name = self.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_display(annotate, save_name=save_name)
        display.show()

    def save_image(self, annotate=True, dpi=300):
        display = self._prepare_display(annotate, self.get_image_save_name())
        save_path = os.path.join(
            self.global_config.IMAGE_SAVE_DIR, self.get_image_save_name()
        )
        display.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        display.close()

    def get_image_save_name(self):
        return self.root_file_name + "_" + self.image_suffix + ".png"

    def save_image_array(self):
        return self._save_image_array(
            self.global_config.ARRAY_SAVE_DIR, self.root_file_name
        )

    def get_image_array_save_name(self):
        return self._get_image_array_save_name(self.root_file_name)

    def save_meta(self):
        return self._save_meta(self.global_config.META_SAVE_DIR, self.root_file_name)

    @property
    def image_suffix(self):
        return "NDVI_image"

    @property
    def root_file_name(self):
        return self._root_file_name

    @property
    def cmap(self):
        return self._cmap

    @property
    def analyzer(self):
        return self._analyzer


# Binned NDVI
# **********************************************


class BinnedNDVIArrayTransformer(Transformer):
    def __init__(self):
        pass

    def apply(self, image_array, bin_edges):
        ndvi_binned = np.digitize(image_array, bin_edges, right=True)
        return ndvi_binned


class BinnedNDVIMetaTransformer(Transformer):
    def __init__(self):
        pass

    def apply(self, metadata, bin_ranges, bin_edges):
        self.metadata = metadata
        self.metadata["band_map"] = {"Binned_NDVI": 1}
        self.metadata["bin_ranges"] = bin_ranges
        self.metadata["bin_edges"] = bin_edges
        return self.metadata


class BinnedNDVITransformer(Transformer):
    def __init__(self):
        self.array_transfomer = BinnedNDVIArrayTransformer()
        self.meta_transfomer = BinnedNDVIMetaTransformer()

    def _calculate_bin_edges(self, bin_ranges):
        bin_edges = [-1]
        [bin_edges.append(bin_ranges[key]) for key in bin_ranges.keys()]

        return sorted(bin_edges)

    def apply(self, ndvi_image, bin_ranges, root_file_name):
        ndvi_image_array = ndvi_image.get_image_array()
        metadata = ndvi_image.get_meta()
        bin_edges = self._calculate_bin_edges(bin_ranges)
        array = self.array_transfomer.apply(ndvi_image_array, bin_edges)

        metadata = self.meta_transfomer.apply(metadata, bin_ranges, bin_edges)

        return BinnedNDVIImage(array, metadata, root_file_name)


class BinnedNDVIAnalyzer:
    def __init__(self, image):
        self.binned_ndvi_image = image

    def _count_pixels(self, image_array):
        unique_values, counts = np.unique(image_array, return_counts=True)
        return unique_values, counts

    def calculate_bin_percentages(self, image_array):
        unique_values, counts = self._count_pixels(image_array)
        percentages = np.round((counts / sum(counts)) * 100, 1)
        return {v: p for v, p in zip(unique_values, percentages)}

    def _prepare_bin_bars(self, annotate, save_name=None, bins=100):
        fig, ax = plt.subplots(figsize=(8, 6))
        custom_cmap = self.binned_ndvi_image._create_custom_cmap(
            self.binned_ndvi_image.cmap,
            self.binned_ndvi_image.get_meta_key("bin_edges"),
        )
        legend_elements = self.binned_ndvi_image._create_legend_elements(
            custom_cmap,
            self.binned_ndvi_image.get_meta_key("bin_ranges"),
            self.binned_ndvi_image.get_meta_key("bin_edges"),
        )

        bin_percentages = self.calculate_bin_percentages(
            self.binned_ndvi_image.get_image_array()
        ).values()

        colors = [v["color"] for v in legend_elements.values()]
        labels = [k for k in legend_elements.keys()]

        areas = plt.bar(labels, bin_percentages, color=colors)

        plt.ylabel("Bin Percentages")
        plt.xticks(rotation=45, ha="right")

        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

        for area, percentage in zip(areas, bin_percentages):
            if area.get_height() < 5:
                label_height = max(area.get_height() * 1.5, 2)
            else:
                label_height = area.get_height() - 2
            plt.text(
                area.get_x() + area.get_width() / 2,
                label_height,
                f"{percentage}%",
                ha="center",
                va="top",
                color="black",
                fontsize=8,
            )

        if annotate:
            ax.set_title("Binned NDVI - Landcover Segments")
            if save_name is not None:
                fig.text(0.5, 0.001, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout(pad=2)
        return plt

    def display_bin_bars(self, annotate=True):
        if annotate:
            save_name = self.binned_ndvi_image.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_bin_bars(annotate, save_name=save_name)
        display.show()


class BinnedNDVIImage(Image):
    def __init__(self, array, metadata, root_file_name, cmap="RdBu_r"):
        Meta.__init__(self, metadata)
        ImageArray.__init__(self, array)
        self.global_config = GlobalConfig()
        self._analyzer = BinnedNDVIAnalyzer(
            self
        )  # Composition: Image contains an Analyzer
        self.save_dir = self.global_config.SAVE_DIR
        self._root_file_name = root_file_name
        self._cmap = cmap

    def _create_custom_cmap(self, cmap, bin_edges):
        image_cmap = plt.get_cmap(cmap)
        custom_cmap = ListedColormap(image_cmap(np.linspace(0, 1, len(bin_edges) - 1)))

        return custom_cmap

    def _create_legend_elements(self, cmap, bin_ranges, bin_edges):
        legend_elements = {}

        for i, label in enumerate(bin_ranges.keys()):
            legend_elements[label] = {
                "id": i + 1,
                "color": cmap(i / (len(bin_edges) - 1)),
            }

        return legend_elements

    def _prepare_display(self, annotate, save_name=None):
        fig, ax = plt.subplots(figsize=(8, 10))
        custom_cmap = self._create_custom_cmap(
            self.cmap, self.get_meta_key("bin_edges")
        )
        legend_elements = self._create_legend_elements(
            custom_cmap, self.get_meta_key("bin_ranges"), self.get_meta_key("bin_edges")
        )

        patches = [Patch(color=v["color"], label=k) for k, v in legend_elements.items()]

        cax = ax.imshow(self.image_array, cmap=custom_cmap)
        plt.legend(
            handles=patches,
            loc="center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=max(1, len(legend_elements) / 2),
        )
        plt.axis("off")

        if annotate:
            ax.set_title("Binned NDVI Image")
            if save_name is not None:
                fig.text(0.5, 0.02, save_name, ha="center", fontsize=6, color="gray")

        plt.tight_layout()
        return plt

    def display(self, annotate=True):
        if annotate:
            save_name = self.get_image_save_name()
        else:
            save_name = None
        display = self._prepare_display(annotate, save_name=save_name)
        display.show()

    def save_image(self, annotate=True, dpi=300):
        display = self._prepare_display(annotate, self.get_image_save_name())
        save_path = os.path.join(
            self.global_config.IMAGE_SAVE_DIR, self.get_image_save_name()
        )
        display.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        display.close()

    def get_image_save_name(self):
        return self.root_file_name + "_" + self.image_suffix + ".png"

    def save_image_array(self):
        return self._save_image_array(
            self.global_config.ARRAY_SAVE_DIR, self.root_file_name
        )

    def get_image_array_save_name(self):
        return self._get_image_array_save_name(self.root_file_name)

    def save_meta(self):
        return self._save_meta(self.global_config.META_SAVE_DIR, self.root_file_name)

    @property
    def image_suffix(self):
        return "binned_NDVI_image"

    @property
    def root_file_name(self):
        return self._root_file_name

    @property
    def cmap(self):
        return self._cmap

    @property
    def analyzer(self):
        return self._analyzer


if __name__ == "__main__":
    pass

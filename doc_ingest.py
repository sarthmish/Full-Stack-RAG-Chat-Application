import logging
from pathlib import Path
from typing import Optional
import tomllib

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    ResponseFormat,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, PictureDescriptionData, PictureMiscData

# --- Logger and Config Setup ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

def parse_toml_file(file_path):
    """Loads a TOML configuration file."""
    try:
        with open(file_path, "rb") as f:
            config = tomllib.load(f)
        logger.info(f"Successfully loaded configuration from {file_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {file_path}")
        raise
    except tomllib.TOMLDecodeError:
        logger.error(f"Error decoding the TOML file: {file_path}")
        raise

config = parse_toml_file("config.toml")
logger.info("Imported and configured DocIngest module")

ARTIFACTS_PATH = config["DOCLING_ARTIFACTS_PATH"]


# --- Main Class Definition ---

class DocIngest:
    """Document ingestion processor for Docling."""

    DEFAULT_IMAGE_SCALE = config.get("DEFAULT_IMAGE_SCALE", 2.0)
    SUPPORTED_EXTENSIONS = {'.json': 1, '.pdf': 0}
    DEFAULT_VLM_ENDPOINT = config.get("DEFAULT_VLM_ENDPOINT")
    DEFAULT_VLM_TIMEOUT = config.get("DEFAULT_VLM_TIMEOUT", 600)

    def __init__(self, source: str):
        self.source = Path(source)
        self.result: Optional[DoclingDocument] = None
        self.json_path: Optional[Path] = None

    def load_file(
        self,
        do_ocr: bool = config["DOCLING_OPTIONS"]["do_ocr"],
        do_table_structure: bool = config["DOCLING_OPTIONS"]["do_table_structure"],
        do_cell_matching: bool = config["DOCLING_OPTIONS"]["do_cell_matching"],
        images_scale: float = DEFAULT_IMAGE_SCALE,
        generate_page_images: bool = config["DOCLING_OPTIONS"]["generate_page_images"],
        generate_picture_images: bool = config["DOCLING_OPTIONS"]["generate_picture_images"],
        do_picture_classification: bool = config["DOCLING_OPTIONS"]["do_picture_classification"],
        do_picture_description: bool = config["DOCLING_OPTIONS"]["do_picture_description"],
        save_visual_grounding: bool = config["DOCLING_OPTIONS"]["save_visual_grounding"],
        vlm_model: str = config["DOCLING_OPTIONS"]["vlm_model"],
        vlm_prompt: str = config["DOCLING_OPTIONS"]["vlm_prompt"],
        vlm_endpoint: str = DEFAULT_VLM_ENDPOINT,
        vlm_timeout: int = DEFAULT_VLM_TIMEOUT,
        output_path: Optional[str] = None
    ) -> None:
        """
        Load and process a document file.

        Returns None if successful, raises Exception if failed.
        """
        logger.info(f"Processing file: {self.source}")

        file_type = self._check_file_type(self.source)
        if file_type == -1:
            logger.error("Unsupported file format. Only JSON and PDF files are accepted.")
            return

        if file_type == 1:  # JSON file
            logger.info("Loading JSON file")
            self.result = DoclingDocument.load_from_json(self.source)
            self.json_path = self.source
            return

        # Process PDF file
        pipeline_options = self._create_pipeline_options(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            do_cell_matching=do_cell_matching,
            images_scale=images_scale,
            generate_page_images=generate_page_images or save_visual_grounding,
            generate_picture_images=generate_picture_images,
            do_picture_classification=do_picture_classification,
            do_picture_description=do_picture_description,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt,
            vlm_endpoint=vlm_endpoint,
            vlm_timeout=vlm_timeout
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        logger.info("Starting document conversion")
        conv_result = doc_converter.convert(self.source)
        logger.info("End of conversion")

        # Build reference lookup for ALL items
        # https://github.com/docling-project/docling/issues/1298
        ref_lookup = {}
        for item in conv_result.document.texts:
            if hasattr(item, 'self_ref'):
                ref_key = f"#{item.self_ref}" if not item.self_ref.startswith('#') else item.self_ref
                ref_lookup[ref_key] = item

        for picture in conv_result.document.pictures:
            description_exists = any(isinstance(item, PictureDescriptionData) for item in picture.annotations)
            if not description_exists:
                picture.annotations.append(
                    PictureDescriptionData(
                        text=f"Text in Picture: {', '.join([ref_lookup.get(item.cref, '').text for item in picture.children if item.cref in ref_lookup])}",
                        provenance=""
                    )
                )
            picture.annotations.append(
                PictureMiscData(
                    content={
                        "captions": ", ".join([ref_lookup.get(item.cref, '').text for item in picture.captions if item.cref in ref_lookup])
                    }
                )
            )

        if output_path:
            json_path = self._generate_output_path(conv_result.input.file, output_path)
            conv_result.document.save_as_json(json_path)
            self.json_path = json_path
            logger.info(f"Saved processed document to: {json_path}")

        self.result = conv_result.document

    @staticmethod
    def _create_pipeline_options(**kwargs) -> PdfPipelineOptions:
        """Create pipeline options with the given parameters."""
        options = PdfPipelineOptions(artifacts_path=ARTIFACTS_PATH)
        options.do_ocr = kwargs.get('do_ocr', True)
        options.do_table_structure = kwargs.get('do_table_structure', True)
        options.table_structure_options.do_cell_matching = kwargs.get('do_cell_matching', True)
        options.ocr_options.lang = ["en"]
        options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.CUDA
        )
        options.images_scale = kwargs.get('images_scale', DocIngest.DEFAULT_IMAGE_SCALE)
        options.generate_page_images = kwargs.get('generate_page_images', True)
        options.generate_picture_images = kwargs.get('generate_picture_images', True)
        options.do_picture_classification = kwargs.get('do_picture_classification', True)
        options.do_picture_description = kwargs.get('do_picture_description', True)
        options.enable_remote_services = True
        options.picture_description_options = DocIngest._create_vlm_options(**kwargs)
        return options

    @staticmethod
    def _create_vlm_options(**kwargs) -> PictureDescriptionApiOptions:
        """Create VLM options with the given parameters."""
        return PictureDescriptionApiOptions(
            url=kwargs.get('vlm_endpoint', DocIngest.DEFAULT_VLM_ENDPOINT),
            params=dict(
                model=kwargs.get('vlm_model', "granite3.2-vision:latest"),
                seed=42,
                max_completion_tokens=500,
            ),
            prompt=kwargs.get('vlm_prompt', ""),
            timeout=kwargs.get('vlm_timeout', DocIngest.DEFAULT_VLM_TIMEOUT),
            scale=1.0,
            response_format=ResponseFormat.MARKDOWN,
        )

    @staticmethod
    def _check_file_type(file_path: Path) -> int:
        """Check file extension and return corresponding type code."""
        return DocIngest.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), -1)

    @staticmethod
    def _generate_output_path(input_file: Path, output_dir: str) -> Path:
        """Generate output JSON path based on input filename."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"document_{input_file.stem.replace(' ', '_')}.json"
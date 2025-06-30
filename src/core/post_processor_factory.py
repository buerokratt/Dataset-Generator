from src.core.post_processor import (
    PostProcessor,
    ZipPostProcessor,
    AggregationPostProcessor,
)


class PostProcessorFactory:
    @staticmethod
    def create_post_processor(config: dict) -> PostProcessor:
        """Create a post-processor based on configuration."""
        post_processing_type = config.get("dataset_generation", {}).get(
            "post_processing", "zip"
        )

        if post_processing_type == "aggregation":
            return AggregationPostProcessor(config)
        elif post_processing_type == "zip":
            return ZipPostProcessor(config)
        else:
            raise ValueError(f"Unknown post-processing type: {post_processing_type}")

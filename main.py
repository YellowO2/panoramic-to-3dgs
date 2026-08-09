from panoramic_to_3dgs import Pipeline, PipelineConfig, load_panorama_folder

if __name__ == "__main__":
    config = PipelineConfig.from_yaml("config.yaml")
    pipeline = Pipeline(config)

    # --- Option A: folder input ---
    # panos, _, _ = load_panorama_folder("data/inputs/panoramas_example")
    # pipeline.run(panos[0], output_dir="data/outputs/folder_test", support_paths=panos[1:])

    # --- Option B: manual target ---
    pipeline.run(
        target_appearance_path="data/inputs/panoramas_sea_view/pano_rTCgvONHkRFIqvygt6llLA.jpg",
        output_dir="data/outputs/multi_pano_test",
    )

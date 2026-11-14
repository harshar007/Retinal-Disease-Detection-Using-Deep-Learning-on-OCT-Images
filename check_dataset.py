from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder

builder = load_dataset_builder("zacharielegault/Kermany2017-OCT")
print("Description:", builder.info.description[:200] if builder.info.description else "No desc")
print("Features:", builder.info.features)
print("Splits:", builder.info.splits)
print("Dataset size:", builder.info.dataset_size)
print("Download size:", builder.info.download_size)

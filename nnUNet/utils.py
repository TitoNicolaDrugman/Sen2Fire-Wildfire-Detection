import kagglehub


def kaggle_download_dataset(name: str):
    path = kagglehub.dataset_download(name)
    print("Dataset downloaded to:", path)
    return path

if __name__ == "__main__":
    kaggle_download_dataset("shariaarfin/sen2fire")
from src.captchadataset import CaptchaDataset

dataset = CaptchaDataset(
    r"C:\Users\Johannes\.cache\kagglehub\datasets\parsasam\captcha-dataset\versions\1"
)
print(len(dataset))
print(dataset[0])


import pandas as pd
import random

# Generate synthetic data for Korean to Chinese translation
num_samples = 100  # You can increase this number based on your needs

# Example Korean (Hangul) and Chinese (Hanzi) characters
korean_samples = ["안녕하세요", "제품", "좋은 아침입니다", "이것은 예시입니다", "어떻게 지내십니까"]
chinese_samples = ["你好", "产品", "早上好", "这是一个例子", "你怎么样"]

# Create a DataFrame
data = {
    "korean": [random.choice(korean_samples) for _ in range(num_samples)],
    "chinese": [random.choice(chinese_samples) for _ in range(num_samples)]
}
df = pd.DataFrame(data)

# Save the DataFrame to CSV files
df.to_csv("path_to_train.csv", index=False)
df[:int(num_samples * 0.1)].to_csv("path_to_validation.csv", index=False)

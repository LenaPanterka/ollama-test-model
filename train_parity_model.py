import json

import ollama

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


MODEL = "llama3.2:latest"


def generate_numbers_with_labels() -> list[dict]:
    prompt = """
Сгенерируй ровно 100 целых чисел.

Для каждого числа правильно определи чётность:
- "even" — если число чётное;
- "odd" — если число нечётное.

Верни ровно 100 объектов.
"""

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 100,
                "maxItems": 100,
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer"
                        },
                        "label": {
                            "type": "string",
                            "enum": ["even", "odd"]
                        }
                    },
                    "required": ["number", "label"]
                }
            }
        },
        "required": ["items"]
    }

    for attempt in range(1, 4):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                format=schema,
                options={
                    "temperature": 0
                },
            )

            content = response["message"]["content"].strip()

            data = json.loads(content)
            items = data["items"]

            if len(items) != 100:
                print(
                    f"Ollama вернула {len(items)} записей вместо 100"
                )
                continue

            return items

        except (json.JSONDecodeError, KeyError, TypeError) as error:
            print(f"Некорректный ответ Ollama: {error}")

    raise ValueError(
        "Не удалось получить 100 корректных записей после 3 попыток"
    )

def train_model(data: list[dict]) -> DecisionTreeClassifier:
    x_train = [[item["number"]] for item in data]
    y_train = [
        0 if item["label"] == "even" else 1
        for item in data
    ]

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    return model

def evaluate_model(model, test_data: list[dict]) -> float:
    x_test = [[item["number"]] for item in test_data]

    y_true = [
        0 if item["label"] == "even" else 1
        for item in test_data
    ]

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy

def generate_test_numbers() -> list[dict]:
    return generate_numbers_with_labels()


def main() -> None:
    data = generate_numbers_with_labels()
    model = train_model(data)

    print("Модель scikit-learn обучена")

    test_data = generate_test_numbers()

    print(f"Получено тестовых записей: {len(test_data)}")

    accuracy = evaluate_model(model, test_data)

    print(f"Точность модели: {accuracy * 100:.0f}%")


if __name__ == "__main__":
    main()
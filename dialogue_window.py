import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ollama import Ollama
except ImportError:
    Ollama = None


class DialogueWindow(QWidget):
    """
    Окно приложения для текстового диалога с локальной языковой моделью через Ollama.
    """

    def __init__(self, model_name: str = "llama3"):
        super().__init__()
        self.setWindowTitle("Textual Converse App")
        self.resize(600, 400)

        self.model_name = model_name
        self.ollama = Ollama() if Ollama else None

        self.init_ui()
        self.dialog_history = []

    def init_ui(self):
        """
        Инициализация интерфейса: поле истории, поле ввода, кнопка отправки.
        """
        font = QFont()
        font.setPointSize(12)

        # История диалога (только для чтения)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(font)

        # Поле ввода запроса
        self.input_line = QLineEdit()
        self.input_line.setFont(font)
        self.input_line.returnPressed.connect(self.send_message)

        # Кнопка отправки
        self.send_button = QPushButton("Отправить")
        self.send_button.setFont(font)
        self.send_button.clicked.connect(self.send_message)

        # Компоновка
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.input_line)
        h_layout.addWidget(self.send_button)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.history_text)
        v_layout.addLayout(h_layout)

        self.setLayout(v_layout)

    def append_to_history(self, speaker: str, text: str):
        """
        Добавляет сообщение в историю диалога с форматированием.

        Args:
            speaker (str): "Пользователь" или "Модель"
            text (str): текст сообщения
        """
        self.dialog_history.append((speaker, text))
        self.update_history_display()

    def update_history_display(self):
        """
        Обновляет отображение всей истории диалога в QTextEdit.
        """
        formatted = ""
        for speaker, text in self.dialog_history:
            formatted += f"[{speaker}]: {text}\n"
        self.history_text.setPlainText(formatted)
        self.history_text.verticalScrollBar().setValue(
            self.history_text.verticalScrollBar().maximum()
        )

    def send_message(self):
        """
        Обрабатывает отправку сообщения пользователем, получение ответа модели,
        обновление истории и очистку поля ввода.
        """
        user_text = self.input_line.text().strip()
        if not user_text:
            return

        self.append_to_history("Пользователь", user_text)
        self.input_line.clear()

        if not self.ollama:
            self.append_to_history(
                "Модель",
                "Ошибка: библиотека ollama не установлена или недоступна.",
            )
            return

        try:
            response = self.ollama.chat(self.model_name, user_text)
            model_reply = response.get("choices", [{}])[0].get("message", "")
            if not model_reply:
                model_reply = "(пустой ответ модели)"
            self.append_to_history("Модель", model_reply)
        except Exception as e:
            self.append_to_history("Модель", f"Ошибка при обращении к модели: {e}")


def main():
    """
    Запуск приложения.
    """
    app = QApplication(sys.argv)
    # Можно изменить модель здесь, например "llama3.2", "gemma2" и т.п.
    model_name = "llama3"
    window = DialogueWindow(model_name=model_name)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

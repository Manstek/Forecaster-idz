import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import base64

from io import BytesIO
from django.shortcuts import render
from .forms import TimeSeriesForm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def generate_dollar_to_ruble_series(n_samples=1000, start_rate=75):
    time = np.arange(0, n_samples)
    trend = 0.01 * time  # Медленный рост курса
    seasonality = 2 * np.sin(2 * np.pi * time / 365)  # Годовая сезонность
    noise = np.random.normal(scale=0.5, size=n_samples)  # Случайные колебания
    series = start_rate + trend + seasonality + noise
    return series


# Подготовка данных
def prepare_data(series, time_steps):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


# Обработка файла
def process_file(file):
    content = file.read().decode('utf-8')
    lines = content.splitlines()
    series = [float(line.strip()) for line in lines if line.strip()]
    return np.array(series)


def index(request):
    form = TimeSeriesForm(request.POST or None, request.FILES or None)
    plot_url = None

    if request.method == "POST" and form.is_valid():
        # Получаем параметры из формы
        n_samples = form.cleaned_data.get('n_samples', 1000)  # Устанавливаем дефолтное значение
        start_rate = form.cleaned_data.get('start_rate', 75.0)  # Устанавливаем дефолтное значение
        time_steps = form.cleaned_data.get('time_steps', 10)  # Устанавливаем дефолтное значение
        random_generate = form.cleaned_data.get('random_generate', False)

        # Получаем временной ряд
        if form.cleaned_data.get('user_series'):  # Если файл загружен
            file = form.cleaned_data['user_series']
            series = process_file(file)
            n_samples = len(series)  # Обновляем количество точек в зависимости от файла
        elif random_generate:  # Если генерация случайного ряда
            series = generate_dollar_to_ruble_series(n_samples, start_rate)
        else:  # Если ничего не выбрано
            series = np.array([float(x) for x in form.cleaned_data.get('user_series', '').split(',')])

        # Нормализация данных
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        # Подготовка данных для LSTM
        X, y = prepare_data(series_scaled, time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Создание и обучение модели
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(time_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        # Прогнозирование
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        y_original = scaler.inverse_transform(y.reshape(-1, 1))

        # Визуализация
        plt.figure(figsize=(12, 6))
        plt.plot(range(time_steps, n_samples), y_original, label='Истинные значения', color='blue')
        plt.plot(range(time_steps, n_samples), predictions, label='Прогноз', color='orange')
        plt.xlabel('Время')
        plt.ylabel('Значения')
        plt.legend()
        plt.title('Прогнозирование временного ряда с помощью LSTM')

        # Сохранение графика в формате base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.read()).decode('utf-8')

    return render(request, 'forecast/index.html', {'form': form, 'plot_url': plot_url})

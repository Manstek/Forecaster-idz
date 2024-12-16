import numpy as np
from django.shortcuts import render
from .forms import TimeSeriesForm
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Генерация временного ряда курса доллара к рублю
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

# Главная страница
def index(request):
    form = TimeSeriesForm(request.POST or None)
    plot_url = None

    if request.method == "POST" and form.is_valid():
        n_samples = form.cleaned_data['n_samples']
        start_rate = form.cleaned_data['start_rate']
        time_steps = form.cleaned_data['time_steps']
        random_generate = form.cleaned_data['random_generate']

        # Генерация временного ряда
        if random_generate:
            series = generate_dollar_to_ruble_series(n_samples, start_rate)
        else:
            # Получение временного ряда из ввода пользователя (например, как список)
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
        plt.ylabel('Курс USD/RUB')
        plt.legend()
        plt.title('Прогнозирование курса доллара к рублю с помощью LSTM')

        # Сохранение графика в формате base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.read()).decode('utf-8')

    return render(request, 'forecast/index.html', {'form': form, 'plot_url': plot_url})

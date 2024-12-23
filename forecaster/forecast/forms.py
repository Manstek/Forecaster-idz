from django import forms


class TimeSeriesForm(forms.Form):
    n_samples = forms.IntegerField(label='Количество точек', required=False)
    start_rate = forms.FloatField(label='Начальный курс', required=False)
    time_steps = forms.IntegerField(label='Шаг времени', required=False)
    random_generate = forms.BooleanField(
        label='Сгенерировать случайный ряд', required=False)
    user_series = forms.FileField(
        label='Загрузите файл с временным рядом', required=False)

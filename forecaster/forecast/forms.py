from django import forms

class TimeSeriesForm(forms.Form):
    n_samples = forms.IntegerField(label='Количество точек данных', min_value=10, initial=1000)
    start_rate = forms.FloatField(label='Начальный курс', min_value=0, initial=75)
    time_steps = forms.IntegerField(label='Шаг времени для прогноза', min_value=1, initial=10)
    random_generate = forms.BooleanField(label='Сгенерировать случайный временной ряд', required=False)


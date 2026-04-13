document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultBox = document.getElementById('result');
    const loading = document.getElementById('loading');
    const errorMsg = document.getElementById('error');
    
    // Адрес вашего API (проверьте порт в терминале запуска)
    const API_URL = '/api/v1/predict';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Скрытие предыдущих результатов
        errorMsg.classList.add('hidden');
        resultBox.classList.remove('hidden');
        loading.classList.remove('hidden');

        // Сбор данных
        const payload = {
            GenHlth: parseInt(document.getElementById('GenHlth').value),
            BMI: parseFloat(document.getElementById('BMI').value),
            Age: parseInt(document.getElementById('Age').value),
            HighBP: parseInt(document.getElementById('HighBP').value),
            HeartDiseaseorAttack: parseInt(document.getElementById('HeartDiseaseorAttack').value),
            DiffWalk: parseInt(document.getElementById('DiffWalk').value),
            PhysActivity: parseInt(document.getElementById('PhysActivity').value)
        };

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Ошибка сервера');
            }

            const data = await response.json();
            displayResult(data);

        } catch (error) {
            console.error('Ошибка:', error);
            loading.classList.add('hidden');
            errorMsg.classList.remove('hidden');
            errorMsg.textContent = 'Не удалось соединиться с сервером. Убедитесь, что запущен uvicorn.';
        }
    });

    function displayResult(data) {
        loading.classList.add('hidden');
        
        // Отображение вероятности и уровня риска
        const percent = (data.probability * 100).toFixed(1);
        document.getElementById('riskPercent').textContent = `${percent}%`;
        
        const riskEl = document.getElementById('riskLevel');
        riskEl.textContent = data.risk_level;
        
        // Цветовое кодирование
        if (percent < 30) riskEl.style.color = '#2D8730'; // Низкий
        else if (percent < 70) riskEl.style.color = '#F5A623'; // Средний
        else riskEl.style.color = '#E25353'; // Высокий

        // Вывод топ-факторов
        const list = document.getElementById('factorsList');
        list.innerHTML = '';
        if (data.top_factors && data.top_factors.length > 0) {
            data.top_factors.forEach(factor => {
                const li = document.createElement('li');
                li.textContent = factor;
                list.appendChild(li);
            });
        } else {
            list.innerHTML = '<li>Нет выраженных факторов</li>';
        }
    }
});
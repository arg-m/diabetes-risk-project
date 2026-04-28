const form = document.getElementById("riskForm");
const result = document.getElementById("result");

const API_URL = "/api/v1/predict";

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  showLoading();

  const payload = {
    GenHlth: num("GenHlth"),
    BMI: num("BMI"),
    Age: num("Age"),
    HighBP: num("HighBP"),
    PhysActivity: num("PhysActivity"),
    HeartDiseaseorAttack: num("HeartDiseaseorAttack"),
    DiffWalk: num("DiffWalk")
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Ошибка API");
    }

    renderResult(data);

  } catch (error) {
    showError(error.message);
  }
});

function num(id) {
  return Number(document.getElementById(id).value);
}

function showLoading() {
  result.classList.remove("hidden");
  result.innerHTML = `
    <div class="result-title">Анализ...</div>
    <div style="opacity:.8;">Модель оценивает риск диабета</div>
  `;
}

function showError(msg) {
  result.classList.remove("hidden");
  result.innerHTML = `
    <div class="error">
      ${msg}
    </div>
  `;
}

function renderResult(data) {
  const percent = Math.round(data.probability * 100);

  let cls = "low";
  if (percent >= 70) cls = "high";
  else if (percent >= 30) cls = "medium";

  const factors = data.top_factors
    .map(x => `<li>${x}</li>`)
    .join("");

  result.classList.remove("hidden");

  result.innerHTML = `
    <div class="result-title">
      Ваш риск: ${percent}%
    </div>

    <div class="result-risk">
      ${data.risk_level}
    </div>

    <div class="progress">
      <div class="progress-fill ${cls}" style="width:${percent}%"></div>
    </div>

    <div>
      <strong>Наибольшее влияние:</strong>
      <ul class="factor-list">
        ${factors}
      </ul>
    </div>

    <div style="margin-top:14px;opacity:.85;">
      ${data.message}
    </div>
  `;
}
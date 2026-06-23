const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

const recommendationForm = document.querySelector("#recommendation-form");
const userInput = document.querySelector("#user-id");
const recommendableUsersFilter = document.querySelector("#recommendable-users-filter");
const userFilterHint = document.querySelector("#user-filter-hint");
const recommendationAlert = document.querySelector("#recommendation-alert");
const recommendationResults = document.querySelector("#recommendation-results");

const sentimentForm = document.querySelector("#sentiment-form");
const sentimentAlert = document.querySelector("#sentiment-alert");
const sentimentResult = document.querySelector("#sentiment-result");


function setActiveTab(tabName) {
    tabButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabName);
    });

    tabPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === `${tabName}-panel`);
    });
}


function showAlert(target, message, type = "error") {
    if (!message) {
        target.innerHTML = "";
        return;
    }

    target.innerHTML = `<div class="alert ${type}">${message}</div>`;
}


function setButtonLoading(button, isLoading, loadingText) {
    if (isLoading) {
        button.dataset.originalText = button.textContent;
        button.textContent = loadingText;
        button.disabled = true;
        return;
    }

    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
}


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function renderRecommendations(recommendations) {
    if (!recommendations || recommendations.length === 0) {
        recommendationResults.innerHTML = "";
        return;
    }

    const cards = recommendations.map((item) => `
        <article class="product-card">
            <div>
                <span class="rank">#${item.rank}</span>
                <h3 class="product-title">${escapeHtml(item.product_name)}</h3>
                <p class="product-id">${escapeHtml(item.product_id)}</p>
            </div>

            <div class="card-stats">
                <div class="stat-row">
                    <span>Positive sentiment</span>
                    <strong>${item.positive_sentiment_percentage.toFixed(2)}%</strong>
                </div>
                <div class="stat-row">
                    <span>Review count</span>
                    <strong>${item.total_review_count}</strong>
                </div>
                <div class="stat-row">
                    <span>Recommendation score</span>
                    <strong>${item.recommendation_score.toFixed(4)}</strong>
                </div>
            </div>
        </article>
    `).join("");

    recommendationResults.innerHTML = `
        <div class="recommendation-grid">
            ${cards}
        </div>
    `;
}


function renderSentiment(prediction) {
    if (!prediction) {
        sentimentResult.innerHTML = "";
        return;
    }

    const sentimentClass = prediction.sentiment.toLowerCase();
    const confidence = prediction.confidence !== null && prediction.confidence !== undefined
        ? `<span class="confidence-pill">${prediction.confidence.toFixed(2)}% confidence</span>`
        : "";

    const ruleNote = prediction.used_rule_override
        ? `<div class="rule-note">Explicit negative wording was detected and handled.</div>`
        : "";

    sentimentResult.innerHTML = `
        <article class="sentiment-card ${sentimentClass}">
            <div class="sentiment-result-line">
                <div>
                    <p class="eyebrow dark">Prediction</p>
                    <div class="sentiment-label">${prediction.sentiment}</div>
                </div>
                ${confidence}
            </div>

            ${ruleNote}

            <p class="review-preview">
                ${escapeHtml(prediction.review_text)}
            </p>
        </article>
    `;
}


tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
        setActiveTab(button.dataset.tab);
    });
});

recommendableUsersFilter.addEventListener("change", () => {
    if (recommendableUsersFilter.checked) {
        userInput.setAttribute("list", "recommendable-user-list");
        userFilterHint.textContent = "Filtered: dropdown now shows only users with valid recommendation scores.";
    } else {
        userInput.setAttribute("list", "all-user-list");
        userFilterHint.textContent = "Showing all users. Enable the filter to show only users with valid recommendation scores.";
    }

    userInput.value = "";
    recommendationResults.innerHTML = "";
    showAlert(recommendationAlert, "");
});


recommendationForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const submitButton = recommendationForm.querySelector("button[type='submit']");
    const formData = new FormData(recommendationForm);

    showAlert(recommendationAlert, "");
    recommendationResults.innerHTML = "";
    setButtonLoading(submitButton, true, "Finding products...");

    try {
        const response = await fetch("/recommend", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.status !== "success") {
            showAlert(recommendationAlert, data.message, data.status);
            return;
        }

        showAlert(recommendationAlert, data.message, "success");
        renderRecommendations(data.recommendations);
    } catch (error) {
        showAlert(recommendationAlert, "Something went wrong while generating recommendations.");
    } finally {
        setButtonLoading(submitButton, false);
    }
});


sentimentForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const submitButton = sentimentForm.querySelector("button[type='submit']");
    const formData = new FormData(sentimentForm);

    showAlert(sentimentAlert, "");
    sentimentResult.innerHTML = "";
    setButtonLoading(submitButton, true, "Predicting...");

    try {
        const response = await fetch("/predict-sentiment", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.status !== "success") {
            showAlert(sentimentAlert, data.message, "error");
            return;
        }

        renderSentiment(data.prediction);
    } catch (error) {
        showAlert(sentimentAlert, "Something went wrong while predicting sentiment.");
    } finally {
        setButtonLoading(submitButton, false);
    }
});

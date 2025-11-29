document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predict-form");
    const messageInput = document.getElementById("message");
    const loadingEl = document.getElementById("loading");
    const errorEl = document.getElementById("error");
    const resultEl = document.getElementById("result");

    const predictionValue = document.getElementById("prediction-value");
    const textProbValue = document.getElementById("text-prob-value");
    const urlProbValue = document.getElementById("url-prob-value");
    const finalProbValue = document.getElementById("final-prob-value");
    const extractedUrlValue = document.getElementById("extracted-url-value");
    const domainValue = document.getElementById("domain-value");

    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        const message = messageInput.value.trim();
        errorEl.style.display = "none";
        resultEl.style.display = "none";

        if (!message) {
            errorEl.textContent = "Please enter a message.";
            errorEl.style.display = "block";
            return;
        }

        loadingEl.style.display = "block";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();
            loadingEl.style.display = "none";

            if (!response.ok) {
                errorEl.textContent = data.error || "Something went wrong.";
                errorEl.style.display = "block";
                return;
            }

            // Fill result fields
            const prediction = data.prediction || "";
            predictionValue.textContent = prediction;
            predictionValue.classList.remove("spam", "ham");
            if (prediction === "SPAM") {
                predictionValue.classList.add("spam");
            } else if (prediction === "HAM") {
                predictionValue.classList.add("ham");
            }

            textProbValue.textContent =
                data.text_spam_probability !== null &&
                data.text_spam_probability !== undefined
                    ? data.text_spam_probability
                    : "N/A";

            urlProbValue.textContent =
                data.url_spam_probability !== null &&
                data.url_spam_probability !== undefined
                    ? data.url_spam_probability
                    : "No URL";

            finalProbValue.textContent =
                data.final_spam_probability !== null &&
                data.final_spam_probability !== undefined
                    ? data.final_spam_probability
                    : "N/A";

            extractedUrlValue.textContent = data.extracted_url || "None";
            domainValue.textContent = data.domain || "None";

            resultEl.style.display = "block";
        } catch (err) {
            loadingEl.style.display = "none";
            errorEl.textContent = "Network / server error. Please try again.";
            errorEl.style.display = "block";
        }
    });
});

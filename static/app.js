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
            const response = await fetch("/api/predict", {
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

            // Fill results
            predictionValue.textContent = data.Prediction || "N/A";
            predictionValue.classList.remove("spam", "ham");

            if (data.Prediction === "SPAM") predictionValue.classList.add("spam");
            if (data.Prediction === "HAM") predictionValue.classList.add("ham");

            textProbValue.textContent = data.Text_Spam_Probability ?? "N/A";
            urlProbValue.textContent = data.URL_Spam_Probability ?? "No URL";
            finalProbValue.textContent = data.Final_Spam_Probability ?? "N/A";
            extractedUrlValue.textContent = data.Extracted_URL || "None";
            domainValue.textContent = data.Domain || "None";

            resultEl.style.display = "block";

        } catch (err) {
            loadingEl.style.display = "none";
            errorEl.textContent = "Network / server error. Please try again.";
            errorEl.style.display = "block";
        }
    });
});

// Store the full API response to avoid re-fetching
let analysisResult = null;

// Helper to format timestamp (e.g., 125 seconds -> "2:05")
function formatTimestamp(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${min}:${sec}`;
}

// Main function to process the video URL
async function processVideo() {
    const url = document.getElementById("videoUrl").value.trim();
    if (!url) {
        alert("Please enter a valid YouTube video URL.");
        return;
    }

    const resultsSection = document.getElementById("results-section");
    const loader = document.getElementById("loader-overlay");
    const loaderText = document.getElementById("loader-text");
    const progressBar = document.getElementById("progress-bar");

    // Show loader and reset state
    loader.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    loaderText.textContent = "Initializing...";
    progressBar.style.width = "0%";
    analysisResult = null; // Clear previous results

    try {
        const response = await fetch("/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n\n");

            lines.forEach(line => {
                if (line.startsWith("data: ")) {
                    const data = JSON.parse(line.substring(6));
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Update progress
                    if (data.status) {
                        loaderText.textContent = data.status;
                    }
                    if (data.progress) {
                        progressBar.style.width = data.progress + "%";
                    }

                    // Handle final result
                    if (data.result) {
                        analysisResult = data.result;
                        populateResults(analysisResult);
                    }
                }
            });
        }

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("summary-card").innerHTML = `<p style="color:#ff4c4c;">❌ Error: ${error.message}</p>`;
        resultsSection.classList.remove("hidden");
    } finally {
        // Hide loader after a short delay to show 100%
        setTimeout(() => loader.classList.add("hidden"), 500);
    }
}

// Function to populate the UI with results
function populateResults(data) {
    const summaryCard = document.getElementById("summary-card");
    const transcriptBox = document.getElementById("transcript-box");
    const momentsCarousel = document.getElementById("moments-carousel");
    const resultsSection = document.getElementById("results-section");
    const toggle = document.getElementById("output-type-toggle");

    if (data) {
        // Populate based on toggle state
        summaryCard.innerHTML = toggle.checked ? data.notes.replace(/\n/g, '<br>') : data.summary;
        transcriptBox.textContent = data.transcript;

        // Populate key moments
        momentsCarousel.innerHTML = "";
        if (data.moments && data.moments.length > 0) {
            data.moments.forEach(moment => {
                const videoId = document.getElementById("videoUrl").value.match(/(?:v=|\/)([0-9A-Za-z_-]{11})/)?.[1];
                const momentLink = videoId ? `https://www.youtube.com/watch?v=${videoId}&t=${moment.timestamp}s` : '#';
                
                const card = document.createElement("div");
                card.className = "moment-card";
                card.innerHTML = `
                    <a href="${momentLink}" target="_blank" class="moment-timestamp">${formatTimestamp(moment.timestamp)}</a>
                    <div class="moment-description">${moment.text}</div>
                `;
                momentsCarousel.appendChild(card);
            });
        } else {
            momentsCarousel.innerHTML = `<p style="color: var(--text-secondary);">Could not extract distinct key moments.</p>`;
        }
        
        resultsSection.classList.remove("hidden");
    }
}

// --- Event Listeners ---

// Listen for the toggle switch change
document.getElementById("output-type-toggle").addEventListener("change", (event) => {
    if (analysisResult) {
        const summaryCard = document.getElementById("summary-card");
        summaryCard.innerHTML = event.target.checked 
            ? analysisResult.notes.replace(/\n/g, '<br>') 
            : analysisResult.summary;
    }
});

// Listen for clicks on copy buttons
document.querySelectorAll('.copy-btn').forEach(button => {
    button.addEventListener('click', () => {
        const targetId = button.dataset.target;
        const contentElement = document.getElementById(targetId);
        
        if (contentElement) {
            navigator.clipboard.writeText(contentElement.innerText).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => { button.textContent = 'Copy'; }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        }
    });
});

// Allow pressing Enter in the input field to trigger the button
document.getElementById("videoUrl").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("analyzeBtn").click();
    }
});
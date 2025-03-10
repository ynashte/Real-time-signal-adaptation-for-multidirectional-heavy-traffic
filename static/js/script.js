function uploadVideo() {
    const videoInput = document.getElementById('videoInput');
    const videoElement = document.getElementById('uploadedVideo');
    const videoContainer = document.getElementById('videoContainer');
    const predictButton = document.getElementById('predictButton');

    if (videoInput.files && videoInput.files[0]) {
        const videoFile = videoInput.files[0];
        const videoURL = URL.createObjectURL(videoFile);

        videoElement.src = videoURL;
        videoContainer.style.display = 'block';
        predictButton.style.display = 'block';
    } else {
        alert("Failed to load video. Please try again.");
    }
}

async function processFrame() {
    const videoElement = document.getElementById('uploadedVideo');
    videoElement.pause();

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    const frameDataUrl = canvas.toDataURL('image/jpeg');

    const response = await fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ frame: frameDataUrl })
    });

    const result = await response.json();

    if (result.error) {
        alert(result.error);
        return;
    }

    document.getElementById('vehicleCount').textContent = `Vehicles Detected: ${result.vehicle_count}`;
    document.getElementById('predictedTime').textContent = `Predicted Time: ${result.predicted_time}`;

    const outputImage = document.getElementById('outputImage');
    outputImage.src = result.output_image + '?t=' + new Date().getTime();
    outputImage.style.display = 'block';

    document.getElementById('results').style.display = 'block';
}

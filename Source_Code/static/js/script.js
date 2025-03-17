// Show Upload Indication
document.getElementById('image-upload').addEventListener('change', function () {
    const uploadIndication = document.getElementById('upload-indication');
    const fileName = this.files[0] ? this.files[0].name : '';

    // Show upload indication
    uploadIndication.textContent = `Uploaded: ${fileName} ✅`;
    uploadIndication.style.display = 'inline';
});
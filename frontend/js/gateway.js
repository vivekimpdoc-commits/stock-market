function updateTime() {
    const now = new Date();
    document.getElementById('sysTime').innerText = now.toISOString().substr(11, 8) + ' UTC';
}
setInterval(updateTime, 1000);
updateTime();

function handleSubmit() {
    const btn = document.getElementById('submitBtn');
    btn.classList.add('loading');
    
    // Handshake simulation
    setTimeout(() => {
        btn.classList.remove('loading');
        localStorage.setItem("gatewayPassed", "true");
        window.location.href = "index.html";
    }, 800);
}

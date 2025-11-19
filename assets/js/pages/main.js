const btnCadastrar = document.querySelector(".main-btn");
const overlay = document.getElementById("modal-overlay");

btnCadastrar.addEventListener("click", () => {
    overlay.style.display = "flex";
});

document.getElementById("cancelar").addEventListener("click", () => {
    overlay.style.display = "none";
});

overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
        overlay.style.display = "none";
    }
});

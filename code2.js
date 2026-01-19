let form = document.querySelector("#profileForm");
let inputs = document.querySelectorAll("input");
let main = document.querySelector("#main");

form.addEventListener("submit",function(dets){
    dets.preventDefault();

    let card = document.createElement("div");
    card.classList.add("card");
    card.classList.add("cards")

    let profile = document.createElement("div");
    profile.classList.add("profilePic")

    let img = document.createElement("img");
    img.setAttribute("src", inputs[3].value);

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
    img.classList.add("img")

    let h3 = document.createElement("h3");
    h3.textContent = inputs[0].value;
    let h4 = document.createElement("h4");
    h4.textContent = inputs[1].value;
    let h5 = document.createElement("h5");
    h5.textContent = inputs[2].value;

    profile.appendChild(img);
    card.appendChild(profile);

    card.appendChild(h3);
    card.appendChild(h4);
    card.appendChild(h5);

    main.appendChild(card);

    inputs.forEach(function(inp){
        if(inp.type!=="submit"){
        inp.value = "";}
    });


});
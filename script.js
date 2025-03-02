// Function to handle project uploads
function uploadProject() {
  const input = document.getElementById("projectUpload");
  const file = input.files[0];

  if (file) {
    const projectList = document.getElementById("project-list");
    const reader = new FileReader();

    reader.onload = function (e) {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.style.width = "100px";
      img.style.margin = "10px";
      projectList.appendChild(img);
    };

    reader.readAsDataURL(file);
  } else {
    alert("Please select an image file to upload.");
  }
}

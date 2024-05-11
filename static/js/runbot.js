// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Get the button element
  var sendButton = document.querySelector(".btn-send");

  // Add click event listener to the button
  sendButton.addEventListener("click", function () {
    // Get the input field value
    var userInput = document.getElementById("user-input").value;

    // Log the input value to the console
    console.log("User input:", userInput);
  });
});

function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  if (userInput.trim() === "") return;

  // Send user input to the Flask backend
  fetch("/send_message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_input: userInput }),
  })
    .then((response) => response.json())
    .then((data) => {
      // Log bot response to console (for testing)
      console.log(data.bot_response);
      // Update UI with bot response (you can customize this part)
      // For example, you can display the bot response in the chat container
      var chatContainer = document.getElementById("chat-container");
      var botMessageElement = document.createElement("div");
      botMessageElement.classList.add("message");
      botMessageElement.textContent = data.bot_response;
      chatContainer.appendChild(botMessageElement);
      // Scroll to the bottom of the chat container
      chatContainer.scrollTop = chatContainer.scrollHeight;
    });

  // Clear the input field
  document.getElementById("user-input").value = "";
}

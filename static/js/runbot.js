// Wait for the DOM to be fully loaded
// document.addEventListener("DOMContentLoaded", function () {
//   // Get the button element
//   var sendButton = document.querySelector(".btn-send");

//   // Add click event listener to the button
//   sendButton.addEventListener("click", function () {
//     // Get the input field value
//     var userInput = document.getElementById("user-input").value;

//     // Log the input value to the console
//     console.log("User input:", userInput);
//   });
// });

document.getElementById("user-input").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
      event.preventDefault();  // Prevent the default action (e.g., form submission)
      sendMessage();  // Call the sendMessage function
  }
});

function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  const urlParts = window.location.pathname.split("/");
  sessionId = urlParts[urlParts.length - 1];

  if (userInput.trim() === "") return;
  var chatContainer = document.getElementById("chat-container");

  // Create a new message element for the user's input
  var userMessageElement = document.createElement("div");
  userMessageElement.classList.add("message", "user-message");
  userMessageElement.textContent = userInput;
  chatContainer.appendChild(userMessageElement);

  // Send user input to the Flask backend
  fetch("/send_message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_input: userInput, session_id: sessionId }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log(data.history_context)
      console.log(data.bot_response);

      if (Array.isArray(data.bot_response)) {
        data.bot_response.forEach((response) => {
          var botMessageElement = document.createElement("div");
          botMessageElement.classList.add("message", "bot-message");
          botMessageElement.textContent = response;
          chatContainer.appendChild(botMessageElement);
        });
      } else {
        // If the response is not an array, handle it as a single message
        var botMessageElement = document.createElement("div");
        botMessageElement.classList.add("message", "bot-message");
        botMessageElement.textContent = data.bot_response;
        chatContainer.appendChild(botMessageElement);
      }
      // Scroll to the bottom of the chat container
      chatContainer.scrollTop = chatContainer.scrollHeight;
    });

  // Clear the input field
  document.getElementById("user-input").value = "";
}

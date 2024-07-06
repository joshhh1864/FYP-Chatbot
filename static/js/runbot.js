document
  .getElementById("user-input")
  .addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent the default action (e.g., form submission)
      sendMessage(); // Call the sendMessage function
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
      console.log(data.history_context);
      console.log(data.bot_response);

      if (Array.isArray(data.bot_response)) {
        data.bot_response.forEach((response) => {
          // var botMessageElement = document.createElement("div");
          // botMessageElement.classList.add("message", "bot-message");
          // botMessageElement.textContent = response;
          // chatContainer.appendChild(botMessageElement);
          appendBotMessage(response);
        });
      } else {
        // If the response is not an array, handle it as a single message
        // var botMessageElement = document.createElement("div");
        // botMessageElement.classList.add("message", "bot-message");
        // botMessageElement.textContent = data.bot_response;
        // chatContainer.appendChild(botMessageElement);
        appendBotMessage(response);
      }
      // Scroll to the bottom of the chat container
      chatContainer.scrollTop = chatContainer.scrollHeight;
    });

  // Clear the input field
  document.getElementById("user-input").value = "";
}

function appendBotMessage(response) {
  var chatContainer = document.getElementById("chat-container");

  // Create a new message element for the bot's response
  var botMessageElement = document.createElement("div");
  botMessageElement.classList.add("message", "bot-message");
  botMessageElement.textContent = response;

  // Create a container for the feedback buttons
  var feedbackContainer = document.createElement("div");
  feedbackContainer.classList.add("feedback-container");

  var feedbackMessage = document.createElement("span");
  feedbackMessage.textContent = "Did you find this helpful?";
  feedbackMessage.classList.add("feedback-message");

  // Create thumbs up button
  var thumbsUp = document.createElement("span");
  thumbsUp.textContent = "ദ്ദി´▽`)";
  thumbsUp.classList.add("feedback-btn", "thumbs-up");
  thumbsUp.onclick = () => handleFeedback(response, 0);

  // Create thumbs down button
  var thumbsDown = document.createElement("span");
  thumbsDown.textContent = "( ,,⩌'︿'⩌,,)";
  thumbsDown.classList.add("feedback-btn", "thumbs-down");
  thumbsDown.onclick = () => handleFeedback(response, 1);

  // Append the buttons to the feedback container
  feedbackContainer.appendChild(feedbackMessage);
  feedbackContainer.appendChild(thumbsUp);
  feedbackContainer.appendChild(thumbsDown);

  // Append the feedback container to the bot message
  botMessageElement.appendChild(feedbackContainer);

  // Append the bot message to the chat container
  chatContainer.appendChild(botMessageElement);
}

function handleFeedback(response, type) {
  const urlParts = window.location.pathname.split("/");
  sessionId = urlParts[urlParts.length - 1];

  console.log(`Feedback for response "${response}": ${type === 0 ? "Positive" : "Negative"}`);
  
  // Optionally send the feedback to the backend
  fetch("/send_feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ response, feedback: type , sessionid: sessionId}),
  });
}

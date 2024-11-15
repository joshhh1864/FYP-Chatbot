document
  .getElementById("user-input")
  .addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent the default action (e.g., form submission)
      sendMessage(); // Call the sendMessage function
    }
  });

function showNotification(message, type = "success") {
  const container = document.getElementById("notification-container");

  // Create notification element
  const notification = document.createElement("div");
  notification.classList.add("notification", type);
  notification.innerHTML = `
      <span>${message}</span>
      <span class="close-btn">&times;</span>
    `;

  // Append notification to container
  container.appendChild(notification);

  // Show notification
  setTimeout(() => {
    notification.classList.add("show");
  }, 10);

  // Remove notification after 3 seconds
  setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => {
      container.removeChild(notification);
    }, 500);
  }, 3000);

  // Add close button event
  notification.querySelector(".close-btn").addEventListener("click", () => {
    notification.classList.remove("show");
    setTimeout(() => {
      container.removeChild(notification);
    }, 500);
  });
}

let feedbackButtonAdded = false;

function sendMessage() {
  feedbackButtonAdded = false;
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
        data?.bot_response
          .filter((response) => response && typeof response === "string")
          .forEach((response) => {
            appendBotMessage(response);
          });
      } else if (data?.bot_response && typeof data.bot_response === "string") {
        appendBotMessage(data.bot_response);
      }
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

  if (!feedbackButtonAdded) {
    var feedbackContainer = document.createElement("div");
    feedbackContainer.classList.add("feedback-container");

    var feedbackMessage = document.createElement("span");
    feedbackMessage.textContent = "Did you find this helpful?";
    feedbackMessage.classList.add("feedback-message");

    // Create thumbs up button
    var thumbsUp = document.createElement("span");
    thumbsUp.textContent = "ദ്ദി´▽`)";
    thumbsUp.classList.add("feedback-btn", "thumbs-up");
    thumbsUp.onclick = () => handleFeedback(botMessageElement, response, 0);

    // Create thumbs down button
    var thumbsDown = document.createElement("span");
    thumbsDown.textContent = "(ᴗ_ ᴗ。) ᴖ̈.";
    thumbsDown.classList.add("feedback-btn", "thumbs-down");
    thumbsDown.onclick = () => handleFeedback(botMessageElement, response, 1);

    // Append the buttons to the feedback container
    feedbackContainer.appendChild(feedbackMessage);
    feedbackContainer.appendChild(thumbsUp);
    feedbackContainer.appendChild(thumbsDown);

    // Append the feedback container to the bot message
    botMessageElement.appendChild(feedbackContainer);

    feedbackButtonAdded = true;
  }

  chatContainer.appendChild(botMessageElement);
}

function handleFeedback(botMessageElement, response, type) {
  const urlParts = window.location.pathname.split("/");
  sessionId = urlParts[urlParts.length - 1];

  console.log(
    `Feedback for response "${response}": ${
      type === 0 ? "Positive" : "Negative"
    }`
  );

  // Optionally send the feedback to the backend
  fetch("/send_feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ response, feedback: type, sessionid: sessionId }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.message === "Feedback recorded successfully.") {
        var feedbackContainer = botMessageElement.querySelector(
          ".feedback-container"
        );
        showNotification("Thank you for your feedback!","success")
        feedbackContainer.style.display = "none";
      } else {
        showNotification("Error in sending feedback","error");
        console.error("Error in sending feedback:", data.error);
      }
    })
    .catch((error) => {
      showNotification("Error in sending feedback","error");
      console.error("Error sending feedback:", error);
    });
}

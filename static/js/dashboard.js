document.addEventListener("DOMContentLoaded", function () {
  function fetchUserInfo() {
    // Call the Flask API endpoint
    fetch("/get_current_user")
      .then((response) => {
        // Check if the response is ok (status code in the range 200-299)
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        // Update the user info in the HTML
        document.getElementById("user-name").textContent = data.username;
      })
      .catch((error) => {
        // Handle any errors that occur during fetch
        console.error(
          "There was an error fetching the user information:",
          error
        );
        document.getElementById("user-name").textContent =
          "Error loading user info";
      });
  }

  fetchUserInfo();

  function handleLogout() {
    fetch("/logout", { method: "POST" })
      .then((response) => {
        if (response.ok) {
          // Redirect to the login page or show a logged-out message
          window.location.href = "/";
        } else {
          return response.json().then((data) => {
            throw new Error(data.error || "Logout failed");
          });
        }
      })
      .catch((error) => {
        console.error("Error during logout:", error);
        alert("Logout failed: " + error.message);
      });
  }

  document.querySelector('.icon-button.logout').addEventListener('click', handleLogout);
});

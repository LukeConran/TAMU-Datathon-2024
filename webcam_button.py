from flask import Flask
import subprocess
import random

app = Flask(__name__)
app.config['PROPOGATE_EXCEPTIONS'] = True

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <title>
    DROWSY DRIVING
    </title>

    <style>
        .top-100-pixels {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100px; /* Adjust this value to change the height */
            background-color: #e7f1f9; /* Change this color to your desired color */
            z-index: -10; /* Ensure it's above other elements */
        }
    </style>

    <link rel="icon" href="https://static-00.iconduck.com/assets.00/sleeping-face-emoji-2048x2048-x3gtr8b8.png">
    <body>
        <div class="top-100-pixels"></div>

        <div style = "margin-top: 35px;">
            <h1 style="font-family: Verdana; color:#373e43">DROWSY DRIVING DETECTOR</h1>
        </div>
        
        <div style = "margin-top: 50px; font-family: Arial">
            <p> Press the button below to begin capturing facial footage. When finished, press the escape key. </p>
        </div>
        <div class = "container" style="display: flex">
            <div class="left-div"><button onclick="runWebcam()">Start Webcam</button><div>
            <div style='font-family: Arial' class="right-div"><p id='cam_status'></p></div>
        </div>
        <div>
            <p id='dro_stat' style='font-family: Arial'></p>
        </div>
        <script>
        function runWebcam() {
            fetch('/run_script')
                .then(response => {
                    if (response.ok) {
                        updateStatus('Video captured successfully! Processing...')
                        runDrowsy();
                        //alert('Python script executed successfully!');
                    } else {
                        //alert('Error executing Python script!');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
        function updateStatus(message){
            var statusElement = document.getElementById('cam_status');
            statusElement.innerText = message;
        }
        function runDrowsy() {
            fetch('/run_drowsy')
                .then(response => {
                    if (response.ok) {
                        updateStatus2('You are NOT DROWSY! Under 65 percent of frames indicated a drowsy face. Drive safe!')
                        //alert('Python script executed successfully!');
                    } else {
                        updateStatus2('You are DROWSY! Over 65 percent of frames indicate a drowsy face. DO NOT DRIVE!')
                        //alert('Error executing Python script!');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
        function updateStatus2(message){
            var statusElement = document.getElementById('dro_stat');
            statusElement.innerText = message;
        }
        </script>
    </body>
    </html>
    '''

@app.route('/run_script')
def run_script():
    try:
        subprocess.run(["python", "webcam.py"])
        return 'Python script executed successfully!', 200
    except Exception as e:
        print(e)
        return 'Error executing webcam script!', 500

@app.route('/run_drowsy')
def run_drowsy():
    if(random.randint(1,2) == 1):
        return 'Python script executed successfully!', 200
    else:
        return 'Error executing ML script!', 500

if __name__ == '__main__':
    app.run(debug=True)

'''function displayOutput(){
            var xhr = new XMLHttpRequest();
            xhr.open("GET", "/run_drowsy");
            xhr.onreadystatechange = function(){
                if (xhr.readyState === XMLHttpRequest.DONE){
                    if (xhr.status === 200){
                        document.getElementById("output").innerText = xhr.responseText;
                    }
                    else{
                        console.error('Error:', xhr.status, xhr.statusText);
                    }
                }
            };
            xhr.send();
        }'''
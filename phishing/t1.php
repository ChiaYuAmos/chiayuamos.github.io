<?php
    echo "test<br>";
    
    $servername = "172.31.2.233:8889";
    $username = "userx";
    $password = "123";
    $dbname = "userx";
    $emailtest = $_POST["email"];
    $pass1 = $_POST["password"];
    echo var_dump($emailtest)."<br>".var_dump($pass1)."<br>";

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    // if ($conn->connect_error) {
    //     die("Connection failed: " . $conn->connect_error);
    // }
    // if ($_SERVER["REQUEST_METHOD"] == "POST") {
    //   echo $_SERVER["REQUEST_METHOD"]."<br>";
    // } else {
    //   echo "没有收到POST请求";
    // }
    $sql = "INSERT INTO account (`id`,`email`,`password`) VALUES (2,'$emailtest', '$pass1')";
    if ($conn->query($sql) === TRUE) {
        echo "Record inserted successfully";
    } else {
        echo "Error: " . $conn->error;
    }
    $conn->close();
?>
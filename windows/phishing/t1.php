<?php
    echo "test";
    
    $servername = "172.29.7.8:3306";
    $username = "uesrx";
    $password = "123";
    $dbname = "uesrx";

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
    $sql = "INSERT INTO account (`id`,`email`,`password`)
            VALUES (2,'testemail', 'testpass')";
    if ($conn->query($sql) === TRUE) {
        echo "Record inserted successfully";
    } else {
        echo "Error: " . $conn->error;
    }
    $conn->close();
?>
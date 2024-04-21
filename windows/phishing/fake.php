<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>PHP與MySQL建立網頁資料庫</title>
</head>
<body>
<?php

if(!empty($_SERVER['HTTP_CLIENT_IP'])){
  $myip = $_SERVER['HTTP_CLIENT_IP'];
}else if(!empty($_SERVER['HTTP_X_FORWARDED_FOR'])){
  $myip = $_SERVER['HTTP_X_FORWARDED_FOR'];
}else{
  $myip= $_SERVER['REMOTE_ADDR'];
}
echo $myip;
echo "<br>test<br>";

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
if(isset($_POST['email'])&& isset($_POST['pass1'] )){
    $YourName = $conn->real_escape_string($_POST['email']);
    $YourPass = $conn->real_escape_string($_POST['pass1']);

    $sql = "INSERT INTO account (`id`,`email`,`password`)
            VALUES (2,'$YourName', '$YourPass')";
  if ($conn->query($sql) === TRUE) {
    echo "Record inserted successfully";
  } else {
    echo "Error: " . $conn->error;
  }
}
else{
  // echo 'pass';
}
$conn->close();
?>
<!-- <meta http-equiv="refresh" content="0;url= https://www.facebook.com/?locale=zh_TW "> -->
</body>
</html>

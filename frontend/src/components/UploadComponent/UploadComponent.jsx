import React, { useState } from "react";
import { Table, Button, Upload, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";

const App = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  const columns = [
    { title: "Rank", dataIndex: "rank", key: "rank" },
    { title: "N-gram", dataIndex: "ngram", key: "ngram" },
    { title: "ƒ", dataIndex: "ƒ", key: "ƒ" }, // Оновлено тут!
    { title: "R", dataIndex: "R", key: "R" },
    { title: "a", dataIndex: "a", key: "a" },
    { title: "b", dataIndex: "b", key: "b" },
    { title: "Goodness", dataIndex: "goodness", key: "goodness" },
];


  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) throw new Error("Upload failed");
      const result = await response.json();
      setData(result);
      message.success("File processed successfully");
    } catch (error) {
      message.error("Error uploading file");
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: 20 }}>
      <Upload
        beforeUpload={(file) => {
          handleUpload(file);
          return false;
        }}
        showUploadList={false}
      >
        <Button icon={<UploadOutlined />} loading={loading}>Upload File</Button>
      </Upload>
      <Table columns={columns} dataSource={data} rowKey="rank" style={{ marginTop: 20 }} />
    </div>
  );
};

export default App;

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDownSampleBlock(nn.Module):
    """A residual block with a 1D convolution for downsampling.
    This block applies two 1D convolutions. The first convolution maintains the
    sequence length, followed by batch normalization and a ReLU activation with a
    residual connection. The second convolution performs downsampling using the
    specified stride.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride value for the second convolution, determining the
            downsampling factor.
        kernel_size (int, optional): Kernel size for the convolutions. Defaults to 4.
    """

    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the residual downsampling block.
        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, out_channels, new_seq_len), where new_seq_len is
                determined by the stride of the second convolution.
        """
        # x: (batch_size, in_channels, seq_len)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output) + x  # Residual Connection
        output = self.conv2(output)
        return output


class DownsamplingNetwork(nn.Module):
    """A network composed of multiple ResidualDownSampleBlocks for downsampling audio features.
    This network first applies mean pooling and then passes the input through a
    series of ResidualDownSampleBlocks, progressively reducing the sequence length.
    Finally, a 1D convolution maps the features to the desired embedding dimension.
    Args:
        embedding_dim (int, optional): The final output embedding dimension.
            Defaults to 128.
        hidden_dim (int, optional): The number of channels used in the
            intermediate ResidualDownSampleBlocks. Defaults to 64.
        in_channels (int, optional): The number of input channels (e.g., 1 for
            raw audio). Defaults to 1.
        initial_mean_pooling_kernel_size (int, optional): Kernel size for the
            initial average pooling layer. Defaults to 2.
        strides (list[int], optional): A list of stride values, one for each
            ResidualDownSampleBlock, determining the downsampling factor at each
            stage. Defaults to [3, 4, 5].
    """

    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=64,
        in_channels=1,
        initial_mean_pooling_kernel_size=2,
        strides=[3, 4, 5],
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.mean_pooling = nn.AvgPool1d(kernel_size=initial_mean_pooling_kernel_size)

        for i in range(len(strides)):
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i > 0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8,
                )
            )
        self.final_conv = nn.Conv1d(
            hidden_dim, embedding_dim, kernel_size=4, padding="same"
        )

    def forward(self, x):
        """Forward pass through the downsampling network.
        Args:
            x (torch.Tensor): Input tensor, typically raw audio or features,
                of shape (batch_size, in_channels, seq_len).
        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, final_seq_len, embedding_dim), where final_seq_len
                is the sequence length after downsampling.
        """
        x = self.mean_pooling(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.final_conv(x)
        x = x.transpose(1, 2)  # Transpose to (batch_size, seq_len, embedding_dim)
        return x


if __name__ == "__main__":
    batch_size = 2
    input_embedding_dim = 1
    seq_len = 237680

    output_embedding_dim = 32
    hidden_dim = 16
    strides = [6, 6, 6]
    initial_mean_pooling_kernel_size = 2

    downsampling_network = DownsamplingNetwork(
        embedding_dim=output_embedding_dim,
        hidden_dim=hidden_dim,
        in_channels=input_embedding_dim,
        initial_mean_pooling_kernel_size=initial_mean_pooling_kernel_size,
        strides=strides,
    )
    x = torch.randn(batch_size, 1, seq_len)
    print(downsampling_network(x).shape)
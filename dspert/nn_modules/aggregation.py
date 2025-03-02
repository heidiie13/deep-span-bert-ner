import torch
from torch import nn

class SequenceGroupAggregating(nn.Module):
    def __init__(self, mode: str = 'mean'):
        """
        Lớp gộp biểu diễn sub-token về token gốc.
        Args:
            mode: Chế độ gộp, hỗ trợ: 'mean', 'max', 'sum', 'first'
        """
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['mean', 'max', 'sum', 'min'], f"Mode {mode} is not supported!"

    def forward(self, hidden: torch.Tensor, ori_indexes: torch.Tensor) -> torch.Tensor:
        """
        Gộp các biểu diễn sub-token thành biểu diễn token gốc.
        Args:
            hidden: Tensor biểu diễn sub-token (shape: [batch_size, num_sub_tokens, hid_dim])
            ori_indexes: Tensor ánh xạ sub-token về token gốc (shape: [batch_size, num_sub_tokens])
        Returns:
            aggregated: Tensor biểu diễn token gốc (shape: [batch_size, num_tokens, hid_dim])
        """
        batch_size, num_sub_tokens, hid_dim = hidden.size()
        device = hidden.device

        # Tìm số lượng token gốc tối đa trong batch
        max_num_tokens = ori_indexes.max().item() + 1  # +1 vì ori_indexes bắt đầu từ 0
        
        # Tạo tensor kết quả
        aggregated = torch.zeros(batch_size, max_num_tokens, hid_dim, device=device)
        
        # Tạo mask để nhóm các sub-token theo token gốc
        mask = torch.zeros(batch_size, num_sub_tokens, max_num_tokens, device=device)
        for b in range(batch_size):
            for s, t in enumerate(ori_indexes[b]):
                if t >= 0:  # Bỏ qua padding (-1)
                    mask[b, s, t] = 1

        if self.mode == 'mean':
            # Tính trung bình các sub-token cho mỗi token gốc
            summed = torch.bmm(mask.transpose(1, 2), hidden)  # [batch_size, max_num_tokens, hid_dim]
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch_size, 1, max_num_tokens]
            aggregated = summed / counts.transpose(1, 2)
        
        elif self.mode == 'max':
            # Lấy giá trị tối đa theo sub-token
            hidden_expanded = hidden.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, hid_dim)
            masked_hidden = hidden_expanded.masked_fill(~mask_expanded.bool(), float('-inf'))
            aggregated, _ = masked_hidden.max(dim=1)
        
        
        elif self.mode == 'min':
            # Lấy giá trị nhỏ nhất theo sub-token
            hidden_expanded = hidden.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, hid_dim)
            masked_hidden = hidden_expanded.masked_fill(~mask_expanded.bool(), float('inf'))
            aggregated, _ = masked_hidden.min(dim=1)
            
        elif self.mode == 'sum':
            # Tính tổng các sub-token
            aggregated = torch.bmm(mask.transpose(1, 2), hidden)

        return aggregated
    

class SequencePooling(nn.Module):
    def __init__(self, mode: str = 'max'):
        """
        Gộp các vector trong chuỗi thành một vector duy nhất.
        Args:
            mode: Phương pháp gộp ('max', 'mean', 'sum')
        """
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['max', 'mean', 'sum'], f"Mode {mode} is not supported!"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào [batch_size, seq_len, hid_dim]
        Returns:
            Tensor đầu ra [batch_size, hid_dim]
        """
        if self.mode == 'max':
            return x.max(dim=1)[0]
        elif self.mode == 'mean':
            return x.mean(dim=1)
        elif self.mode == 'sum':
            return x.sum(dim=1)
        

class SequenceAttention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int = None, num_heads: int = 1, 
                 scoring: str = 'dot', drop_rate: float = 0.0, external_query: bool = False):
        """
        Gộp chuỗi bằng attention với các mode khác nhau, không sử dụng mask.
        Args:
            key_dim: Kích thước của key (và value)
            query_dim: Kích thước của query (mặc định bằng key_dim)
            num_heads: Số đầu attention (mặc định 1)
            scoring: Phương pháp tính điểm ('dot', 'scaled_dot', 'multiplicative', 'additive')
            drop_rate: Tỷ lệ dropout cho trọng số attention
            external_query: Sử dụng query bên ngoài hay query nội bộ (tham số học)
        """
        super().__init__()
        if query_dim is None:
            query_dim = key_dim
        
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.scoring = scoring.lower()
        self.drop_rate = drop_rate
        self.external_query = external_query

        assert self.scoring in ['dot', 'scaled_dot', 'multiplicative', 'additive'], f"Scoring {scoring} is not supported!"

        if not self.external_query:
            self.query = nn.Parameter(torch.empty(query_dim))
            nn.init.normal_(self.query, mean=0, std=0.02)  # Khởi tạo tham số query
            
        if self.scoring == 'multiplicative':
            self.proj_layer = nn.Linear(key_dim // num_heads, query_dim // num_heads, bias=False)
            nn.init.xavier_uniform_(self.proj_layer.weight)  # Khởi tạo tuyến tính
            
        elif self.scoring == 'additive':
            self.proj_layer = nn.Linear((key_dim + query_dim) // num_heads, key_dim // num_heads, bias=False)
            self.w2 = nn.Parameter(torch.empty(key_dim // num_heads))
            nn.init.normal_(self.w2, mean=0, std=0.02)  # Khởi tạo vector w2
            
        self.dropout = nn.Dropout(drop_rate)

    def compute_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        Tính điểm attention dựa trên query và key, không sử dụng mask.
        Args:
            query: Tensor [batch_size, query_step, query_dim]
            key: Tensor [batch_size, key_step, key_dim]
        Returns:
            scores: Tensor [batch_size * num_heads, query_step, key_step]
        """
        # Chuẩn bị multi-head
        if self.num_heads > 1:
            query = self._prepare_multiheads(query)
            key = self._prepare_multiheads(key)

        batch_size, query_step, _ = query.size()
        _, key_step, _ = key.size()

        if self.scoring == 'dot':
            scores = query.bmm(key.permute(0, 2, 1))  # [batch_size * num_heads, query_step, key_step]
        
        elif self.scoring == 'scaled_dot':
            scores = query.bmm(key.permute(0, 2, 1)) / (self.query_dim / self.num_heads) ** 0.5  # Chia cho sqrt(dim)
        
        elif self.scoring == 'multiplicative':
            key_projed = self.proj_layer(key)  # [batch_size * num_heads, key_step, query_dim / num_heads]
            scores = query.bmm(key_projed.permute(0, 2, 1))  # [batch_size * num_heads, query_step, key_step]
        
        elif self.scoring == 'additive':
            # Nối query và key theo chiều cuối
            key_query = torch.cat([key.unsqueeze(1).expand(-1, query_step, -1, -1), 
                                 query.unsqueeze(2).expand(-1, -1, key_step, -1)], dim=-1)
            key_query_projed = self.proj_layer(key_query)  # [batch_size * num_heads, query_step, key_step, key_dim / num_heads]
            scores = key_query_projed.matmul(self.w2)  # [batch_size * num_heads, query_step, key_step]

        return scores

    def _prepare_multiheads(self, x: torch.Tensor):
        """
        Chuẩn bị tensor cho multi-head attention.
        Args:
            x: Tensor [batch_size, step, dim]
        Returns:
            Tensor [batch_size * num_heads, step, dim / num_heads]
        """
        batch_size = x.size(0)
        dim_per_head = x.size(-1) // self.num_heads
        return x.view(batch_size, -1, self.num_heads, dim_per_head).permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, dim_per_head)

    def _restore_multiheads(self, x: torch.Tensor):
        """
        Khôi phục tensor sau multi-head attention.
        Args:
            x: Tensor [batch_size * num_heads, step, dim / num_heads]
        Returns:
            Tensor [batch_size, step, dim]
        """
        batch_size = x.size(0) // self.num_heads
        dim_per_head = x.size(-1)
        return x.view(batch_size, self.num_heads, -1, dim_per_head).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, dim_per_head * self.num_heads)

    def forward(self, x: torch.Tensor, query: torch.Tensor = None, key: torch.Tensor = None, return_atten_weight: bool = False):
        """
        Thực hiện attention trên chuỗi, không sử dụng mask.
        Args:
            x: Tensor giá trị [batch_size, key_step, value_dim]
            query: Tensor query [batch_size, query_step, query_dim] (nếu external_query = True)
            key: Tensor key [batch_size, key_step, key_dim] (nếu khác x)
            return_atten_weight: Trả về cả trọng số attention nếu True
        Returns:
            Tensor giá trị attention [batch_size, query_step, value_dim] hoặc tuple (values, weights)
        """
        # Chuẩn bị query
        if self.external_query:
            assert query is not None, "Query phải được cung cấp khi external_query = True"
        else:
            assert query is None, "Query không cần khi external_query = False"
            query = self.query.expand(x.size(0), 1, -1)  # [batch_size, 1, query_dim]

        # Sử dụng x làm key và value nếu không chỉ định
        if key is None:
            key = x

        # Tính điểm attention
        scores = self.compute_scores(query, key)

        # Chuẩn hóa trọng số attention
        atten_weight = torch.nn.functional.softmax(scores, dim=-1)  # [batch_size * num_heads, query_step, key_step]
        atten_weight = self.dropout(atten_weight)

        # Tính giá trị attention
        if self.num_heads > 1:
            x = self._prepare_multiheads(x)  # [batch_size * num_heads, key_step, value_dim / num_heads]
        atten_values = atten_weight.bmm(x)  # [batch_size * num_heads, query_step, value_dim / num_heads]

        # Khôi phục nếu multi-head
        if self.num_heads > 1:
            atten_values = self._restore_multiheads(atten_values)  # [batch_size, query_step, value_dim]
            if return_atten_weight:
                atten_weight = self._restore_multiheads(atten_weight)  # [batch_size, query_step, key_step]

        # Nếu query ban đầu là vector 1D, squeeze chiều query_step
        if query.size(1) == 1:
            atten_values = atten_values.squeeze(1)  # [batch_size, value_dim]
            if return_atten_weight:
                atten_weight = atten_weight.squeeze(1)  # [batch_size, key_step]

        if return_atten_weight:
            return atten_values, atten_weight
        return atten_values

# # Ví dụ sử dụng
# x = torch.randn(2, 5, 64)  # [batch_size, seq_len, hid_dim]
# for mode in ['dot', 'scaled_dot', 'multiplicative', 'additive']:
#     attn = SequenceAttention(key_dim=64, query_dim=64, num_heads=1, scoring=mode, drop_rate=0.1, external_query=True)
#     output = attn(x, query=x, key=x)
#     print(f"{mode}: {output.shape}")
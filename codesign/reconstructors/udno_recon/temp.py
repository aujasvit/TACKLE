        # (B, C, X, Y, 2)
        recon_zf = complex_to_chan(ifftn_(masked_kspace, dim=[-2, -1]), chan_dim=1, num_chan=1)
        num_low_frequencies = self.acs_ratio

        if(recon_zf.dim() == 5):
            # multi-coil
            recon_zf = _rss(recon_zf, dim=2)
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        else:
            # single coil
            sens_maps = None

        masked_kspace = torch.view_as_real(masked_kspace)
        # reduce before inpainting
        if self.reduction_method == "rss":
            # (B, 1, H, W, 2) single channel image space
            x_reduced = sens_reduce(masked_kspace, sens_maps)
            # (B, 1, H, W, 2)
            k_reduced = _fft2c(x_reduced)
        elif self.reduction_method == "batch":
            k_reduced, b = chans_to_batch_dim(masked_kspace)
        # print(k_reduced.shape, masked_kspace.shape)

        kspace_pred = k_reduced.clone()

        # inpainting
        if self.skip_method == "replace":
            kspace_pred = self.kno(k_reduced)
        elif self.skip_method == "add_inv":
            # FIXME: this is not correct (mask has shape B, 1, H, W, 2 and self.gno(k_reduced) has shape B*C, 1, H, W, 2)
            kspace_pred = k_reduced.clone() + (~mask * self.kno(k_reduced))
        elif self.skip_method == "add":
            kspace_pred = k_reduced.clone() + self.kno(k_reduced)
        elif self.skip_method == "concat":
            kspace_pred = torch.cat([k_reduced.clone(), self.kno(k_reduced)], dim=1)
        else:
            raise NotImplementedError("skip_method not implemented")
        # expand after inpainting
        if self.reduction_method == "rss":
            if self.skip_method == "concat":
                # kspace_pred is (B, 2, H, W, 2)
                kspace = kspace_pred[:, :1, :, :, :]
                in_kspace = kspace_pred[:, 1:, :, :, :]
                # B, 2C, H, W, 2
                kspace_pred = torch.cat(
                    [sens_expand(kspace, sens_maps), sens_expand(in_kspace, sens_maps)],
                    dim=1,
                )
            else:
                # (B, 1, H, W, 2) -> (B, C, H, W, 2) multi-channel k space
                kspace_pred = sens_expand(kspace_pred, sens_maps)
        elif self.reduction_method == "batch":
            # (B, C, H, W, 2) multi-channel k space
            if self.skip_method == "concat":
                kspace = kspace_pred[:, :1, :, :, :]
                in_kspace = kspace_pred[:, 1:, :, :, :]
                # B, 2C, H, W, 2
                kspace_pred = torch.cat(
                    [
                        batch_chans_to_chan_dim(kspace, b),
                        batch_chans_to_chan_dim(in_kspace, b),
                    ],
                    dim=1,
                )
            else:
                kspace_pred = batch_chans_to_chan_dim(kspace_pred, b)

        # iterative update
        # print(masked_kspace.shape, masked_kspace.dtype)
        for cascade in self.cascades:
            kspace_pred = cascade(
                kspace_pred, masked_kspace, mask, sens_maps, self.use_dc_term
            )

        spatial_pred = _ifft2c(kspace_pred)
        spatial_pred_abs = _complex_abs(spatial_pred)
        recon = _rss(spatial_pred_abs, dim=1)
        recon = recon.unsqueeze(dim=0)
        return recon, recon_zf
